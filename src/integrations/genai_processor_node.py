import logging
from typing import Any, Dict

try:
    from genai_processors import content_api, processors, streams
except ImportError:
    processors = None
    content_api = None
    streams = None

logger = logging.getLogger(__name__)

class GenAIProcessorNode:
    """
    A LangGraph-compatible node that wraps a Google GenAI Processor.

    This node adapts the stream-based GenAI Processor API to the state-based
    LangGraph API.
    """

    def __init__(self, processor_instance: Any):
        """
        Initialize with a GenAI Processor instance.

        Args:
            processor_instance: An instance of a class inheriting from genai_processors.Processor
        """
        if processors is None:
            logger.warning("genai-processors library not installed. Node will fail at runtime.")
            self.processor = None
        else:
            self.processor = processor_instance

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the state using the GenAI Processor.

        Expects 'content' or 'input_text' in state.
        Returns 'processed_output' in state.
        """
        if self.processor is None:
            raise ImportError("genai-processors not installed")

        logger.info(f"Executing GenAI Processor: {self.processor.__class__.__name__}")

        # 1. Adapt Input: Convert State -> Processor Stream
        input_data = state.get("content") or state.get("input_text") or "Empty Input"

        # Handle string input
        if isinstance(input_data, str):
            input_parts = [input_data]
        elif isinstance(input_data, list):
            input_parts = input_data
        else:
            input_parts = [str(input_data)]

        # Create stream
        input_stream = streams.stream_content(input_parts)

        # 2. Execute Processor
        output_buffer = []
        try:
            async for part in self.processor(input_stream):
                # Extract text or content from part
                if hasattr(part, 'text'):
                    output_buffer.append(part.text)
                elif hasattr(part, 'content'):
                    output_buffer.append(str(part.content))
                else:
                    output_buffer.append(str(part))
        except Exception as e:
            logger.error(f"Processor execution failed: {e}")
            raise

        # 3. Adapt Output: Stream -> State
        result_text = "".join(output_buffer)

        return {
            "processed_output": result_text,
            "processor_stats": {
                "name": self.processor.__class__.__name__,
                "parts_processed": len(output_buffer)
            }
        }

def create_genai_node(processor_class, **kwargs) -> GenAIProcessorNode:
    """Factory to create a node from a Processor class."""
    if processors is None:
        return GenAIProcessorNode(None)

    instance = processor_class(**kwargs)
    return GenAIProcessorNode(instance)
