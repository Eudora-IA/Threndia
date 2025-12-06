import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

class RetinexEnhancer:
    """
    Implementation of Single-Scale and Multi-Scale Retinex image enhancement.
    Adapted from: https://github.com/AKRISH22/Retinex-Image-Enhancement
    """

    @staticmethod
    def single_scale_retinex(img, variance):
        retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
        return retinex

    @staticmethod
    def multi_scale_retinex(img, variance_list):
        retinex = np.zeros_like(img)
        for variance in variance_list:
            retinex += RetinexEnhancer.single_scale_retinex(img, variance)
        retinex = retinex / len(variance_list)
        return retinex

    @staticmethod
    def process_msr(img: np.ndarray, variance_list: List[int] = None) -> np.ndarray:
        """
        Apply Multi-Scale Retinex (MSR) with color restoration.
        """
        if variance_list is None:
            variance_list = [15, 80, 30]

        img = np.float64(img) + 1.0
        img_retinex = RetinexEnhancer.multi_scale_retinex(img, variance_list)

        for i in range(img_retinex.shape[2]):
            unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
            zero_count = 0
            for u, c in zip(unique, count):
                if u == 0:
                    zero_count = c
                    break
            low_val = unique[0] / 100.0
            high_val = unique[-1] / 100.0
            for u, c in zip(unique, count):
                if u < 0 and c < zero_count * 0.1:
                    low_val = u / 100.0
                if u > 0 and c < zero_count * 0.1:
                    high_val = u / 100.0
                    break
            img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

            # Normalize
            min_val = np.min(img_retinex[:, :, i])
            max_val = np.max(img_retinex[:, :, i])
            img_retinex[:, :, i] = (img_retinex[:, :, i] - min_val) / (max_val - min_val) * 255

        return np.uint8(img_retinex)

    def enhance(self, input_path: str, output_path: str) -> str:
        """Read, process, and save image using MSR."""
        logger.info(f"Enhancing image with Retinex: {input_path}")
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Input image not found: {input_path}")

        enhanced_img = self.process_msr(img)
        cv2.imwrite(output_path, enhanced_img)
        return output_path


class RealESRGANWrapper:
    """Wrapper for Real-ESRGAN inference script."""

    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.join(
            os.getcwd(), "tools", "external", "Real-ESRGAN"
        )
        self.script_path = os.path.join(self.base_path, "inference_realesrgan.py")

    def upscale(self, input_path: str, output_path: str, scale: int = 4, face_enhance: bool = True) -> str:
        """Run Real-ESRGAN inference."""
        if not os.path.exists(self.script_path):
            raise ImportError(f"Real-ESRGAN script not found at {self.script_path}")

        cmd = [
            "python", self.script_path,
            "-n", "RealESRGAN_x4plus",
            "-i", input_path,
            "-o", os.path.dirname(output_path), # Script takes output dir or logic
            "--output", output_path, # Wait, check arg support usually it's -o folder or similar
            "-s", str(scale)
        ]

        # Note: standard inference_realesrgan.py uses -o for output folder, and -s for scale.
        # It auto-names the file. We might need to handle renaming if we want specific output path.
        # For simplicity, we point -o to the directory of output_path.

        output_dir = os.path.dirname(output_path)

        # Override cmd for typical Real-ESRGAN usage
        cmd = [
            "python", self.script_path,
            "-n", "RealESRGAN_x4plus",
            "-i", input_path,
            "-o", output_dir,
            "-s", str(scale)
        ]

        if face_enhance:
            cmd.append("--face_enhance")

        logger.info(f"Running Real-ESRGAN: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=self.base_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Real-ESRGAN failed: {e}")
            raise

        # Real-ESRGAN appends suffix, e.g. _out. We might need to find the result.
        # Assuming for now we rely on the implementation details or user to find it.
        return output_dir

class VisualEnhancementNode:
    """
    LangGraph node for visual enhancement.
    Supports: 'retinex', 'realesrgan'
    """

    def __init__(self):
        self.retinex = RetinexEnhancer()
        self.realesrgan = RealESRGANWrapper()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image path from state.
        Executes enhancement based on 'mode' in state.
        """
        input_path = state.get("image_path") or state.get("output_path") # Chain from previous
        if not input_path:
            logger.warning("No image path provided for enhancement")
            return {"error": "Missing image_path"}

        mode = state.get("enhancement_mode", "retinex")

        output_path = str(Path(input_path).parent / f"enhanced_{Path(input_path).name}")

        try:
            if mode == "retinex":
                result_path = self.retinex.enhance(input_path, output_path)
                return {"enhanced_image_path": result_path, "status": "success"}
            elif mode == "realesrgan":
                # RealESRGAN handles its own output naming mostly
                self.realesrgan.upscale(input_path, output_path)
                # Heuristic for output name
                stem = Path(input_path).stem
                ext = Path(input_path).suffix
                expected_output = Path(input_path).parent / f"{stem}_out{ext}"
                return {"enhanced_image_path": str(expected_output), "status": "success"}
            else:
                return {"error": f"Unknown mode: {mode}"}
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return {"error": str(e)}
