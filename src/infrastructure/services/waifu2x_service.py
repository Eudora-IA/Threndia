"""
Waifu2x integration node.
Wraps local Waifu2x-Extension-GUI or compatible CLI tools.
"""
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.base_node import BaseNode, NodeResult, log_execution

logger = logging.getLogger(__name__)

class Waifu2xNode(BaseNode):
    """
    Waifu2x Node for high-quality upscaling.

    Expects Waifu2x-Extension-GUI or waifu2x-ncnn-vulkan
    to be available locally.
    """

    def __init__(
        self,
        name: str = "waifu2x",
        executable_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        super().__init__(name, config)
        self.executable_path = executable_path or self._find_executable()

    def _find_executable(self) -> Optional[str]:
        """Try to locate Waifu2x executable."""
        # Common paths or environment variables could be checked here
        # For now, we assume it might be in a 'tools' subdir or provided via config
        candidates = [
            "tools/Waifu2x-Extension-GUI/Waifu2x-Extension-GUI-Console.exe",
            "tools/Waifu2x/waifu2x-ncnn-vulkan.exe",
            "Waifu2x-Extension-GUI-Console.exe"
        ]
        for path in candidates:
            if Path(path).exists():
                return path
        return None

    def validate(self) -> bool:
        """Check if Waifu2x is available."""
        if not self.executable_path:
            return False
        return Path(self.executable_path).exists()

    @log_execution
    def execute(self, input_data: Dict[str, Any]) -> NodeResult:
        """
        Upscale image using Waifu2x.

        Args:
            input_data: Dict with:
                - image_path: Input image
                - output_path: Output path (optional)
                - scale: 2 or 4 (default 2)
                - denoise: 0-3 (default 1)
                - model: specific model name if supported
        """
        import time
        start_time = time.perf_counter()

        try:
            image_path = input_data.get("image_path")
            if not image_path:
                return NodeResult(success=False, error="No image path provided")

            input_p = Path(image_path)
            scale = int(input_data.get("scale", 2))
            denoise = int(input_data.get("denoise", 1))

            output_path = input_data.get(
                "output_path",
                str(input_p.parent / f"{input_p.stem}_w2x_x{scale}{input_p.suffix}")
            )

            if not self.executable_path:
                raise RuntimeError("Waifu2x executable not found")

            # Command structure depends on the specific CLI tool.
            # This follows a generic waifu2x-ncnn-vulkan / CPP style:
            # -i input -o output -s scale -n noise
            cmd = [
                self.executable_path,
                "-i", str(image_path),
                "-o", str(output_path),
                "-s", str(scale),
                "-n", str(denoise),
            ]

            # If using Waifu2x-Extension-GUI console, args might vary slightly.
            # Assuming standard flags for now.

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if process.returncode != 0:
                raise RuntimeError(f"Waifu2x failed: {process.stderr}")

            execution_time = time.perf_counter() - start_time

            return NodeResult(
                success=True,
                data={
                    "output_path": output_path,
                    "scale": scale,
                    "denoise": denoise
                },
                metadata={
                    "tool": "waifu2x",
                    "command": " ".join(cmd)
                },
                execution_time=execution_time
            )

        except Exception as e:
            self.logger.error(f"Waifu2x execution failed: {e}")
            return NodeResult(
                success=False,
                error=str(e),
                execution_time=time.perf_counter() - start_time
            )
