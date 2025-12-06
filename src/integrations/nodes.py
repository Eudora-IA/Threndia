"""
ComfyUI integration node for image generation.
Supports CLI and API modes with workflow automation.
"""
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.base_node import BaseNode, NodeResult, log_execution, retry
from ..core.exceptions import IntegrationError

logger = logging.getLogger(__name__)


class ComfyUINode(BaseNode):
    """
    ComfyUI integration for image generation.

    Supports:
    - Flux2.dev models (FP8 for RTX 4090/3090)
    - Workflow JSON execution
    - CLI and API modes
    - Queue management

    Example:
        >>> node = ComfyUINode(api_url="http://localhost:8188")
        >>> result = node.execute({
        ...     "prompt": "anime girl, cyberpunk",
        ...     "width": 1024,
        ...     "height": 1024
        ... })
    """

    def __init__(
        self,
        name: str = "comfyui",
        api_url: str = "http://localhost:8188",
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__(name, config)
        self.api_url = api_url.rstrip("/")
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self._client = None
        self._workflows: Dict[str, Dict] = {}

    def validate(self) -> bool:
        """Check if ComfyUI is accessible."""
        try:
            import httpx
            response = httpx.get(f"{self.api_url}/system_stats", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"ComfyUI not accessible: {e}")
            return False

    def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(timeout=120.0)
            except ImportError:
                raise IntegrationError("httpx not installed")
        return self._client

    def load_workflow(self, workflow_path: str, name: Optional[str] = None) -> str:
        """Load a workflow JSON file."""
        path = Path(workflow_path)
        if not path.exists():
            raise FileNotFoundError(f"Workflow not found: {workflow_path}")

        with open(path) as f:
            workflow = json.load(f)

        name = name or path.stem
        self._workflows[name] = workflow
        self.logger.info(f"Loaded workflow: {name}")
        return name

    def _build_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        model: str = "flux2-dev",
    ) -> Dict:
        """Build a basic txt2img workflow prompt."""
        # This is a simplified workflow structure
        # Real workflows should be loaded from JSON
        return {
            "prompt": {
                "3": {
                    "class_type": "KSampler",
                    "inputs": {
                        "seed": seed if seed >= 0 else int(time.time()),
                        "steps": steps,
                        "cfg": cfg_scale,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": 1.0,
                    }
                },
                "6": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": prompt,
                    }
                },
                "7": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "text": negative_prompt,
                    }
                },
                "5": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {
                        "width": width,
                        "height": height,
                        "batch_size": 1,
                    }
                },
            }
        }

    @log_execution
    @retry(max_attempts=3, delay=2.0)
    def execute(self, input_data: Dict[str, Any]) -> NodeResult:
        """
        Execute image generation.

        Args:
            input_data: Dict with:
                - prompt: Text prompt
                - negative_prompt: Negative prompt (optional)
                - width/height: Image dimensions
                - workflow: Workflow name or dict (optional)

        Returns:
            NodeResult with generated image paths
        """
        import time
        start_time = time.perf_counter()

        try:
            client = self._get_client()

            prompt = input_data.get("prompt", "")
            if not prompt:
                return NodeResult(
                    success=False,
                    error="No prompt provided"
                )

            # Use workflow or build prompt
            if "workflow" in input_data:
                workflow_data = self._workflows.get(input_data["workflow"], {})
            else:
                workflow_data = self._build_prompt(
                    prompt=prompt,
                    negative_prompt=input_data.get("negative_prompt", ""),
                    width=input_data.get("width", 1024),
                    height=input_data.get("height", 1024),
                    steps=input_data.get("steps", 20),
                    cfg_scale=input_data.get("cfg_scale", 7.0),
                    seed=input_data.get("seed", -1),
                )

            # Queue the prompt
            response = client.post(
                f"{self.api_url}/prompt",
                json={"prompt": workflow_data.get("prompt", workflow_data)},
            )
            response.raise_for_status()

            prompt_id = response.json().get("prompt_id")
            self.logger.info(f"Queued prompt: {prompt_id}")

            # Wait for completion
            images = self._wait_for_completion(prompt_id, client)

            execution_time = time.perf_counter() - start_time

            return NodeResult(
                success=True,
                data={
                    "images": images,
                    "prompt_id": prompt_id,
                },
                metadata={
                    "prompt": prompt,
                    "model": input_data.get("model", "default"),
                },
                execution_time=execution_time,
            )

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return NodeResult(
                success=False,
                error=str(e),
                execution_time=time.perf_counter() - start_time,
            )

    def _wait_for_completion(
        self,
        prompt_id: str,
        client,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ) -> List[str]:
        """Wait for prompt completion and return image paths."""
        start = time.time()

        while time.time() - start < timeout:
            response = client.get(f"{self.api_url}/history/{prompt_id}")

            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    images = []

                    for node_id, output in outputs.items():
                        if "images" in output:
                            for img in output["images"]:
                                images.append(img.get("filename", ""))

                    if images:
                        return images

            time.sleep(poll_interval)

        raise TimeoutError(f"Generation timed out after {timeout}s")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        try:
            client = self._get_client()
            response = client.get(f"{self.api_url}/queue")
            return response.json()
        except Exception as e:
            return {"error": str(e)}


class Anime4KNode(BaseNode):
    """
    Anime4K upscaling node.

    Uses pyanime4k for real-time anime upscaling.
    """

    def __init__(
        self,
        name: str = "anime4k",
        scale: float = 2.0,
        config: Optional[Dict] = None,
    ):
        super().__init__(name, config)
        self.scale = scale
        self._upscaler = None

    def validate(self) -> bool:
        """Check if Anime4K is available."""
        try:
            import pyanime4k
            return True
        except ImportError:
            self.logger.warning("pyanime4k not installed")
            return False

    def _get_upscaler(self):
        """Initialize upscaler."""
        if self._upscaler is None:
            try:
                from pyanime4k import ac
                self._upscaler = ac.AC(
                    initGPUCNN=True,
                    CNNType=ac.CNNType.ACNetHDNL3,
                )
            except Exception as e:
                self.logger.warning(f"GPU upscaler failed, using CPU: {e}")
                from pyanime4k import ac
                self._upscaler = ac.AC(initGPU=False)
        return self._upscaler

    @log_execution
    def execute(self, input_data: Dict[str, Any]) -> NodeResult:
        """
        Upscale image using Anime4K.

        Args:
            input_data: Dict with:
                - image_path: Path to input image
                - output_path: Path for output (optional)
                - scale: Upscale factor (optional)
        """
        import time
        start_time = time.perf_counter()

        try:
            image_path = input_data.get("image_path")
            if not image_path:
                return NodeResult(success=False, error="No image path provided")

            scale = input_data.get("scale", self.scale)

            # Determine output path
            input_path = Path(image_path)
            output_path = input_data.get(
                "output_path",
                str(input_path.parent / f"{input_path.stem}_x{int(scale)}{input_path.suffix}")
            )

            upscaler = self._get_upscaler()
            upscaler.loadImage(str(image_path))
            upscaler.process()
            upscaler.saveImage(str(output_path))

            execution_time = time.perf_counter() - start_time

            return NodeResult(
                success=True,
                data={
                    "output_path": str(output_path),
                    "scale": scale,
                },
                metadata={
                    "input_path": str(image_path),
                },
                execution_time=execution_time,
            )

        except Exception as e:
            self.logger.error(f"Upscaling failed: {e}")
            return NodeResult(
                success=False,
                error=str(e),
                execution_time=time.perf_counter() - start_time,
            )


class FireCrawlNode(BaseNode):
    """
    FireCrawl web scraping node.

    Scrapes web content for reference gathering.
    """

    def __init__(
        self,
        name: str = "firecrawl",
        api_key: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__(name, config)
        self.api_key = api_key
        self._client = None

    def validate(self) -> bool:
        """Check if FireCrawl is available."""
        try:
            from firecrawl import FirecrawlApp
            return self.api_key is not None
        except ImportError:
            self.logger.warning("firecrawl-py not installed")
            return False

    def _get_client(self):
        """Get FireCrawl client."""
        if self._client is None:
            from firecrawl import FirecrawlApp
            self._client = FirecrawlApp(api_key=self.api_key)
        return self._client

    @log_execution
    @retry(max_attempts=2)
    def execute(self, input_data: Dict[str, Any]) -> NodeResult:
        """
        Scrape web content.

        Args:
            input_data: Dict with:
                - url: URL to scrape
                - mode: "scrape" or "crawl"
        """
        import time
        start_time = time.perf_counter()

        try:
            url = input_data.get("url")
            if not url:
                return NodeResult(success=False, error="No URL provided")

            client = self._get_client()
            mode = input_data.get("mode", "scrape")

            if mode == "crawl":
                result = client.crawl_url(
                    url,
                    params={"limit": input_data.get("limit", 10)}
                )
            else:
                result = client.scrape_url(url)

            execution_time = time.perf_counter() - start_time

            return NodeResult(
                success=True,
                data=result,
                metadata={"url": url, "mode": mode},
                execution_time=execution_time,
            )

        except Exception as e:
            self.logger.error(f"Scraping failed: {e}")
            return NodeResult(
                success=False,
                error=str(e),
                execution_time=time.perf_counter() - start_time,
            )


class NvidiaDenoiseNode(BaseNode):
    """
    NVIDIA OptiX AI denoiser node.

    Uses OptiX for AI-accelerated denoising.
    """

    def __init__(
        self,
        name: str = "nvidia_denoise",
        denoiser_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__(name, config)
        self.denoiser_path = denoiser_path or "Denoiser.exe"

    def validate(self) -> bool:
        """Check if denoiser is available."""
        return Path(self.denoiser_path).exists()

    @log_execution
    def execute(self, input_data: Dict[str, Any]) -> NodeResult:
        """
        Denoise image using NVIDIA OptiX.

        Args:
            input_data: Dict with:
                - image_path: Input image path
                - output_path: Output path (optional)
                - strength: Denoise strength (optional)
        """
        import time
        start_time = time.perf_counter()

        try:
            image_path = input_data.get("image_path")
            if not image_path:
                return NodeResult(success=False, error="No image path provided")

            input_path = Path(image_path)
            output_path = input_data.get(
                "output_path",
                str(input_path.parent / f"{input_path.stem}_denoised{input_path.suffix}")
            )

            # Run OptiX denoiser
            cmd = [
                self.denoiser_path,
                "-i", str(image_path),
                "-o", str(output_path),
            ]

            if input_data.get("strength"):
                cmd.extend(["-blend", str(input_data["strength"])])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Denoiser failed: {result.stderr}")

            execution_time = time.perf_counter() - start_time

            return NodeResult(
                success=True,
                data={"output_path": output_path},
                metadata={"input_path": str(image_path)},
                execution_time=execution_time,
            )

        except Exception as e:
            self.logger.error(f"Denoising failed: {e}")
            return NodeResult(
                success=False,
                error=str(e),
                execution_time=time.perf_counter() - start_time,
            )
