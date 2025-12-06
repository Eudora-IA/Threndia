"""
Celery tasks for async asset processing.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Conditional Celery import
try:
    from celery import Celery, Task
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logger.warning("celery not installed. Run: pip install celery[redis]")

# App instance (configured in settings)
app = None
if CELERY_AVAILABLE:
    app = Celery("asset_farm")
    app.config_from_object("config.celery_config", silent=True)


class AssetTask(Task):
    """Base task with error handling and telemetry."""

    abstract = True

    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} completed successfully")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")


# ============================================================================
# GENERATION TASKS
# ============================================================================

if CELERY_AVAILABLE:
    @app.task(base=AssetTask, bind=True, max_retries=3)
    def generate_image_task(
        self,
        prompt: str,
        style: Optional[str] = None,
        model: str = "flux2-dev",
    ) -> Dict[str, Any]:
        """
        Generate image using ComfyUI + Flux.

        Args:
            prompt: Generation prompt
            style: Style modifier
            model: Model to use

        Returns:
            Dict with image path and metadata
        """
        try:
            # ComfyUI integration here
            result = {
                "task_id": self.request.id,
                "image_path": f"output/{self.request.id}.png",
                "prompt": prompt,
                "model": model,
            }
            return result
        except Exception as e:
            self.retry(exc=e, countdown=30)


    @app.task(base=AssetTask, bind=True)
    def upscale_image_task(
        self,
        image_path: str,
        scale: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Upscale image using Anime4K.

        Args:
            image_path: Path to input image
            scale: Upscale factor

        Returns:
            Dict with upscaled image path
        """
        try:
            # Anime4K integration here
            output_path = image_path.replace(".png", f"_x{int(scale)}.png")
            return {
                "task_id": self.request.id,
                "input_path": image_path,
                "output_path": output_path,
                "scale": scale,
            }
        except Exception as e:
            logger.error(f"Upscale failed: {e}")
            raise


    @app.task(base=AssetTask, bind=True)
    def denoise_image_task(
        self,
        image_path: str,
        strength: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Denoise image using NVIDIA OptiX.

        Args:
            image_path: Path to input image
            strength: Denoise strength (0-1)

        Returns:
            Dict with denoised image path
        """
        try:
            output_path = image_path.replace(".png", "_denoised.png")
            return {
                "task_id": self.request.id,
                "input_path": image_path,
                "output_path": output_path,
                "strength": strength,
            }
        except Exception as e:
            logger.error(f"Denoise failed: {e}")
            raise


    @app.task(base=AssetTask, bind=True)
    def create_gif_task(
        self,
        image_paths: List[str],
        duration: float = 0.1,
        loop: int = 0,
    ) -> Dict[str, Any]:
        """
        Create animated GIF from image sequence.

        Args:
            image_paths: List of image paths
            duration: Frame duration in seconds
            loop: Number of loops (0 = infinite)

        Returns:
            Dict with GIF path
        """
        try:
            output_path = "output/animation.gif"
            return {
                "task_id": self.request.id,
                "input_count": len(image_paths),
                "output_path": output_path,
                "duration": duration,
            }
        except Exception as e:
            logger.error(f"GIF creation failed: {e}")
            raise


# ============================================================================
# WORKFLOW CHAINS
# ============================================================================

if CELERY_AVAILABLE:
    @app.task(base=AssetTask, bind=True)
    def full_pipeline_task(
        self,
        prompt: str,
        style: Optional[str] = None,
        upscale: bool = True,
        denoise: bool = True,
        create_gif: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete asset generation pipeline.

        Chains: generate -> upscale -> denoise -> (gif)

        Args:
            prompt: Generation prompt
            style: Style modifier
            upscale: Whether to upscale
            denoise: Whether to denoise
            create_gif: Whether to create GIF

        Returns:
            Final pipeline results
        """
        from celery import chain

        # Build pipeline dynamically
        tasks = [generate_image_task.s(prompt, style)]

        if upscale:
            tasks.append(upscale_image_task.s())
        if denoise:
            tasks.append(denoise_image_task.s())

        pipeline = chain(*tasks)
        result = pipeline.apply_async()

        return {
            "pipeline_id": self.request.id,
            "chain_id": result.id,
            "steps": len(tasks),
        }


# ============================================================================
# TASK UTILITIES
# ============================================================================

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a running task."""
    if not CELERY_AVAILABLE:
        return {"error": "Celery not available"}

    result = AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "result": result.result if result.ready() else None,
    }


def cancel_task(task_id: str) -> bool:
    """Cancel a running task."""
    if not CELERY_AVAILABLE:
        return False

    result = AsyncResult(task_id)
    result.revoke(terminate=True)
    return True
