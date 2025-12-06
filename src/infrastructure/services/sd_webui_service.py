"""
Stable Diffusion WebUI (Automatic1111) integration node.
"""
import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.base_node import BaseNode, NodeResult, log_execution, retry
from ..core.exceptions import IntegrationError

logger = logging.getLogger(__name__)

class SDWebUINode(BaseNode):
    """
    Stable Diffusion WebUI (Automatic1111) integration.

    Supports txt2img generation via API.
    """

    def __init__(
        self,
        name: str = "sd_webui",
        api_url: str = "http://127.0.0.1:7860",
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        super().__init__(name, config)
        self.api_url = api_url.rstrip("/")
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def validate(self) -> bool:
        """Check if SD WebUI is accessible."""
        try:
            import httpx
            # /sdapi/v1/options is a good check
            response = httpx.get(f"{self.api_url}/sdapi/v1/options", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"SD WebUI not accessible at {self.api_url}: {e}")
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

    @log_execution
    @retry(max_attempts=3, delay=2.0)
    def execute(self, input_data: Dict[str, Any]) -> NodeResult:
        """
        Execute generation via SD WebUI.

        Args:
            input_data: Dict with:
                - prompt: Text prompt
                - negative_prompt: Negative prompt
                - steps: Sampling steps
                - width/height: Dimensions
                - cfg_scale: CFG Scale
                - sampler_name: Sampler (e.g. "Euler a")
                - model: Checkpoint name (optional, requires switching logic if supported)

        Returns:
            NodeResult with generated image paths
        """
        start_time = time.perf_counter()

        try:
            client = self._get_client()

            prompt = input_data.get("prompt", "")
            if not prompt:
                return NodeResult(success=False, error="No prompt provided")

            payload = {
                "prompt": prompt,
                "negative_prompt": input_data.get("negative_prompt", ""),
                "steps": input_data.get("steps", 20),
                "width": input_data.get("width", 512), # A1111 defaults often 512, but we can req more
                "height": input_data.get("height", 512),
                "cfg_scale": input_data.get("cfg_scale", 7.0),
                "sampler_name": input_data.get("sampler_name", "Euler a"),
                "seed": input_data.get("seed", -1),
                "batch_size": input_data.get("batch_size", 1),
            }

            # Override if model switching logic was implemented, but for now just generation

            self.logger.info(f"Sending request to SD WebUI: {payload['prompt'][:50]}...")
            response = client.post(f"{self.api_url}/sdapi/v1/txt2img", json=payload)
            response.raise_for_status()

            r_json = response.json()
            images_b64 = r_json.get("images", [])

            saved_paths = []

            # Save images
            timestamp = int(time.time())
            for idx, img_data in enumerate(images_b64):
                img_bytes = base64.b64decode(img_data)
                filename = f"sd_gen_{timestamp}_{idx}.png"
                file_path = self.output_dir / filename

                with open(file_path, "wb") as f:
                    f.write(img_bytes)

                saved_paths.append(str(file_path))

            execution_time = time.perf_counter() - start_time

            return NodeResult(
                success=True,
                data={
                    "images": saved_paths,
                    "info": r_json.get("info", ""),
                    "parameters": r_json.get("parameters", {})
                },
                metadata={
                    "prompt": prompt,
                    "generator": "automatic1111"
                },
                execution_time=execution_time
            )

        except Exception as e:
            self.logger.error(f"SD WebUI Generation failed: {e}")
            return NodeResult(
                success=False,
                error=str(e),
                execution_time=time.perf_counter() - start_time
            )
