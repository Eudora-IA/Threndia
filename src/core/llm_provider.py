"""
Local LLM integration via MCP (Model Context Protocol).
Supports Gemma 3 27B with thinking, coding, and tool calling.
"""
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMRole(Enum):
    """Message roles for LLM conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""
    role: LLMRole
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ToolDefinition:
    """Definition of a callable tool for the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None

    def to_dict(self) -> Dict:
        """Convert to OpenAI-style function definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    thinking: Optional[str] = None  # Chain-of-thought
    tool_calls: List[Dict] = field(default_factory=list)
    finish_reason: str = "stop"
    tokens_used: int = 0


# ============================================================================
# BASE LLM PROVIDER
# ============================================================================

class BaseLLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        **kwargs
    ):
        """Stream response tokens."""
        pass


# ============================================================================
# LOCAL LLM VIA OLLAMA/LLAMA.CPP
# ============================================================================

class LocalLLMProvider(BaseLLMProvider):
    """
    Local LLM provider using Ollama or llama.cpp server.

    Supports Gemma 3 27B and other local models with:
    - Chain-of-thought reasoning
    - Tool/function calling
    - Streaming responses

    Example:
        >>> llm = LocalLLMProvider(model="gemma3:27b")
        >>> response = llm.generate([Message(LLMRole.USER, "Hello")])
    """

    def __init__(
        self,
        model: str = "gemma3:27b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        enable_thinking: bool = True,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self._client = None

    def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(timeout=120.0)
            except ImportError:
                self._client = "urllib"
        return self._client

    def _format_messages(self, messages: List[Message]) -> List[Dict]:
        """Format messages for Ollama API."""
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        return formatted

    def _build_system_prompt(self, tools: Optional[List[ToolDefinition]] = None) -> str:
        """Build system prompt with tool definitions."""
        base_prompt = """You are an intelligent AI assistant with the ability to think step-by-step and use tools.

When solving complex problems:
1. First, wrap your reasoning in <thinking>...</thinking> tags
2. Then provide your response or tool calls

"""
        if tools:
            tool_docs = "Available tools:\n"
            for tool in tools:
                tool_docs += f"\n- {tool.name}: {tool.description}\n"
                tool_docs += f"  Parameters: {json.dumps(tool.parameters, indent=2)}\n"

            base_prompt += tool_docs
            base_prompt += """
To call a tool, use this format:
<tool_call>
{"name": "tool_name", "arguments": {...}}
</tool_call>
"""

        return base_prompt

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from local LLM."""
        client = self._get_client()

        # Inject system prompt with tools
        if tools or self.enable_thinking:
            system_prompt = self._build_system_prompt(tools)
            messages = [Message(LLMRole.SYSTEM, system_prompt)] + messages

        formatted_messages = self._format_messages(messages)

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        try:
            if isinstance(client, str):  # urllib fallback
                import urllib.request
                req = urllib.request.Request(
                    f"{self.base_url}/api/chat",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read())
            else:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            content = data.get("message", {}).get("content", "")

            # Extract thinking
            thinking = None
            if "<thinking>" in content and "</thinking>" in content:
                start = content.find("<thinking>") + len("<thinking>")
                end = content.find("</thinking>")
                thinking = content[start:end].strip()
                content = content[end + len("</thinking>"):].strip()

            # Extract tool calls
            tool_calls = []
            if "<tool_call>" in content:
                import re
                matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
                for match in matches:
                    try:
                        tool_calls.append(json.loads(match.strip()))
                    except json.JSONDecodeError:
                        pass
                # Remove tool call markup from content
                content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()

            return LLMResponse(
                content=content,
                thinking=thinking,
                tool_calls=tool_calls,
                tokens_used=data.get("eval_count", 0),
            )

        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def stream(self, messages: List[Message], **kwargs):
        """Stream response tokens."""
        client = self._get_client()
        formatted_messages = self._format_messages(messages)

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
        }

        try:
            if hasattr(client, "stream"):
                with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line)
                            yield data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"Error: {e}"


# ============================================================================
# GOOGLE GEMINI PROVIDER
# ============================================================================

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API provider.

    Example:
        >>> llm = GeminiProvider(api_key="...")
        >>> response = llm.generate([Message(LLMRole.USER, "Hello")])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
    ):
        self.api_key = api_key
        self.model = model
        self._client = None
        self._initialize()

    def _initialize(self):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
            logger.info(f"Gemini initialized with model={self.model}")
        except ImportError:
            logger.warning("google-generativeai not installed")
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from Gemini."""
        if not self._client:
            return LLMResponse(content="Gemini not available", finish_reason="error")

        try:
            # Convert messages to Gemini format
            chat = self._client.start_chat()

            for msg in messages:
                if msg.role == LLMRole.USER:
                    response = chat.send_message(msg.content)

            return LLMResponse(
                content=response.text,
                tokens_used=response.usage_metadata.total_token_count if hasattr(response, "usage_metadata") else 0,
            )
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def stream(self, messages: List[Message], **kwargs):
        """Stream response tokens."""
        if not self._client:
            yield "Gemini not available"
            return

        try:
            response = self._client.generate_content(
                messages[-1].content,
                stream=True,
            )
            for chunk in response:
                yield chunk.text
        except Exception as e:
            yield f"Error: {e}"


# ============================================================================
# LLM MANAGER
# ============================================================================

class LLMManager:
    """
    Manages multiple LLM providers and tool execution.

    Example:
        >>> manager = LLMManager()
        >>> manager.register_tool(ToolDefinition(...))
        >>> response = manager.chat("Generate an image of a cat")
    """

    def __init__(self, default_provider: Optional[BaseLLMProvider] = None):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.tools: Dict[str, ToolDefinition] = {}
        self.default_provider = default_provider
        self.logger = logging.getLogger("llm_manager")

    def add_provider(self, name: str, provider: BaseLLMProvider):
        """Register an LLM provider."""
        self.providers[name] = provider
        if self.default_provider is None:
            self.default_provider = provider

    def register_tool(self, tool: ToolDefinition):
        """Register a callable tool."""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")

    def execute_tool(self, name: str, arguments: Dict) -> Any:
        """Execute a registered tool."""
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")

        tool = self.tools[name]
        if tool.handler is None:
            raise ValueError(f"Tool {name} has no handler")

        self.logger.info(f"Executing tool: {name}")
        return tool.handler(**arguments)

    def chat(
        self,
        message: str,
        provider_name: Optional[str] = None,
        with_tools: bool = True,
    ) -> LLMResponse:
        """
        Chat with the LLM, automatically executing tool calls.

        Args:
            message: User message
            provider_name: Specific provider to use
            with_tools: Whether to include tool definitions

        Returns:
            LLM response with potential tool execution results
        """
        provider = self.providers.get(provider_name) if provider_name else self.default_provider
        if not provider:
            raise RuntimeError("No LLM provider available")

        messages = [Message(LLMRole.USER, message)]
        tools = list(self.tools.values()) if with_tools else None

        response = provider.generate(messages, tools=tools)

        # Execute tool calls if present
        if response.tool_calls:
            tool_results = []
            for call in response.tool_calls:
                try:
                    result = self.execute_tool(call["name"], call.get("arguments", {}))
                    tool_results.append({
                        "tool": call["name"],
                        "result": result,
                    })
                except Exception as e:
                    tool_results.append({
                        "tool": call["name"],
                        "error": str(e),
                    })

            # Add tool results to response
            response.content += f"\n\nTool Results: {json.dumps(tool_results, indent=2)}"

        return response
