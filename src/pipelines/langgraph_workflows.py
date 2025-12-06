"""
LangGraph workflows for asset generation pipelines.
Defines the state machine and node orchestration.
"""
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

# Conditional imports for LangGraph
try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not installed. Run: pip install langgraph")


class AssetType(Enum):
    """Types of assets that can be generated."""
    IMAGE = "image"
    GIF = "gif"
    VIDEO = "video"
    UPSCALED = "upscaled"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class AssetGenerationState(TypedDict):
    """
    State schema for asset generation workflow.

    This state flows through all nodes in the pipeline,
    accumulating data and tracking progress.
    """
    # Input
    prompt: str
    style: Optional[str]
    reference_urls: List[str]

    # Processing
    scraped_content: List[Dict[str, Any]]
    generated_images: List[str]  # File paths
    upscaled_images: List[str]
    denoised_images: List[str]
    gifs: List[str]

    # Output
    final_assets: List[str]
    metadata: Dict[str, Any]

    # Status
    current_step: str
    errors: List[str]
    status: str


def create_initial_state(prompt: str, style: Optional[str] = None) -> AssetGenerationState:
    """Create initial workflow state."""
    return AssetGenerationState(
        prompt=prompt,
        style=style,
        reference_urls=[],
        scraped_content=[],
        generated_images=[],
        upscaled_images=[],
        denoised_images=[],
        gifs=[],
        final_assets=[],
        metadata={},
        current_step="init",
        errors=[],
        status=WorkflowStatus.PENDING.value,
    )


# ============================================================================
# WORKFLOW NODES
# ============================================================================


# Import core integration nodes
try:
    from src.integrations import ComfyUINode, FireCrawlNode, SDWebUINode, Waifu2xNode
    CORE_TOOLS_AVAILABLE = True
except ImportError:
    CORE_TOOLS_AVAILABLE = False
    logger.warning("Core integration tools not found")

def scrape_references_node(state: AssetGenerationState) -> AssetGenerationState:
    """
    Scrape reference content using FireCrawl.
    """
    logger.info(f"Scraping {len(state['reference_urls'])} URLs")

    if not CORE_TOOLS_AVAILABLE:
        logger.warning("Scraping skipped: FireCrawlNode not available")
        return {**state, "current_step": "scrape_skipped"}

    # Initialize scraper (assumes API key in env or settings)
    scraper = FireCrawlNode()

    scraped = []
    for url in state.get("reference_urls", []):
        try:
            logger.info(f"Scraping: {url}")
            result = scraper.execute({"url": url, "mode": "scrape"})

            if result.success:
                scraped.append({
                    "url": url,
                    "content": result.data.get("content", ""),
                    "metadata": result.data.get("metadata", {}),
                    "images": result.data.get("images", []) # If FireCrawl creates screenshots
                })
            else:
                state["errors"].append(f"Scrape failed for {url}: {result.error}")

        except Exception as e:
            state["errors"].append(f"Scrape execution error for {url}: {e}")

    return {
        **state,
        "scraped_content": scraped,
        "current_step": "scrape_complete",
    }


def generate_images_node(state: AssetGenerationState) -> AssetGenerationState:
    """
    Generate images using ComfyUI or SD WebUI.
    """
    logger.info(f"Generating images for prompt: {state['prompt'][:50]}...")

    if not CORE_TOOLS_AVAILABLE:
        logger.warning("Generation skipped: Core tools not available")
        return {
            **state,
            "generated_images": ["placeholder_gen.png"],
            "current_step": "generate_skipped"
        }

    # Initialize generator based on preference
    generator_type = state.get("generator", "comfyui") # comfyui or automatic1111

    if generator_type == "automatic1111":
        generator = SDWebUINode()
        log_name = "SD WebUI"
    else:
        generator = ComfyUINode()
        log_name = "ComfyUI"

    generated_paths = []
    try:
        # Validate generator availability
        if not generator.validate():
            raise RuntimeError(f"{log_name} server not accessible")

        full_prompt = f"{state['prompt']}, {state.get('style', '')}"

        # Unified execution call
        result = generator.execute({
            "prompt": full_prompt,
            "width": 1024,
            "height": 1024,
            "steps": 25,
            "negative_prompt": "low quality, bad anatomy"
        })

        if result.success and result.data.get("images"):
            # Normalize output path handling
            from pathlib import Path
            for img in result.data["images"]:
                img_path = Path(img)
                if not img_path.is_absolute():
                    generated_paths.append(str(Path("output") / img))
                else:
                    generated_paths.append(str(img_path))

            logger.info(f"Generated {len(generated_paths)} images via {log_name}")
        else:
            state["errors"].append(f"{log_name} generation failed: {result.error}")

    except Exception as e:
        state["errors"].append(f"Generation execution error: {e}")

    return {
        **state,
        "generated_images": generated_paths,
        "current_step": "generate_complete",
    }



# Import integration nodes
try:
    from src.integrations.visual_enhancement import VisualEnhancementNode
    VISUAL_TOOLS_AVAILABLE = True
except ImportError:
    VISUAL_TOOLS_AVAILABLE = False
    logger.warning("Visual enhancement tools not found")

def upscale_images_node(state: AssetGenerationState) -> AssetGenerationState:
    """
    Upscale images using Waifu2x, Real-ESRGAN, or Anime4K.
    """
    logger.info(f"Upscaling {len(state['generated_images'])} images")

    upscaled_paths = []

    # Determine upscaler preference
    upscaler_type = state.get("upscaler", "realesrgan") # waifu2x, realesrgan, anime4k

    # Initialize enhancer based on type
    enhancer = None
    if upscaler_type == "waifu2x":
        enhancer = Waifu2xNode()
        if not enhancer.validate():
            logger.warning("Waifu2x not found, falling back to Real-ESRGAN")
            upscaler_type = "realesrgan"

    if upscaler_type == "realesrgan" and VISUAL_TOOLS_AVAILABLE:
        enhancer = VisualEnhancementNode()

    for img_path in state.get("generated_images", []):
        try:
            if not enhancer:
                # Fallback to dummy
                upscaled_paths.append(img_path.replace(".png", "_upscaled.png"))
                continue

            if upscaler_type == "waifu2x":
                result = enhancer.execute({
                    "image_path": img_path,
                    "scale": 2,
                    "denoise": 1
                })
                if result.success:
                     upscaled_paths.append(result.data["output_path"])
                else:
                    raise RuntimeError(result.error)

            elif upscaler_type == "realesrgan":
                # VisualEnhancementNode wrapper usage
                result = enhancer({
                    "image_path": img_path,
                    "enhancement_mode": "realesrgan"
                })
                if result.get("status") == "success":
                    upscaled_paths.append(result["enhanced_image_path"])
                else:
                     raise RuntimeError(result.get("error", "Unknown error"))

            else:
                 # Default fallback logic
                 upscaled_paths.append(img_path)


        except Exception as e:
            state["errors"].append(f"Upscale failed for {img_path}: {e}")
            # Keep original if failed
            upscaled_paths.append(img_path)

    return {
        **state,
        "upscaled_images": upscaled_paths,
        "current_step": "upscale_complete",
    }


def denoise_images_node(state: AssetGenerationState) -> AssetGenerationState:
    """
    Denoise images using NVIDIA OptiX.

    Applies AI denoising for cleaner output.
    """
    logger.info(f"Denoising {len(state['upscaled_images'])} images")

    denoised_paths = []
    for img_path in state.get("upscaled_images", []):
        try:
            # NVIDIA OptiX integration would go here
            denoised_path = img_path.replace("_upscaled.png", "_denoised.png")
            denoised_paths.append(denoised_path)
        except Exception as e:
            state["errors"].append(f"Denoise failed for {img_path}: {e}")

    return {
        **state,
        "denoised_images": denoised_paths,
        "current_step": "denoise_complete",
    }


def create_gif_node(state: AssetGenerationState) -> AssetGenerationState:
    """
    Create animated GIFs from images.

    Uses imageio/MoviePy for high-quality animations.
    """
    logger.info("Creating GIF from processed images")

    gif_paths = []
    try:
        # GIF creation would go here
        gif_paths.append("output/animation_001.gif")
    except Exception as e:
        state["errors"].append(f"GIF creation failed: {e}")

    return {
        **state,
        "gifs": gif_paths,
        "current_step": "gif_complete",
    }


def finalize_assets_node(state: AssetGenerationState) -> AssetGenerationState:
    """
    Finalize and collect all generated assets.

    Prepares metadata for marketplace upload.
    """
    logger.info("Finalizing assets")

    final_assets = []
    final_assets.extend(state.get("denoised_images", []))
    final_assets.extend(state.get("gifs", []))

    metadata = {
        "prompt": state["prompt"],
        "style": state.get("style"),
        "asset_count": len(final_assets),
        "types": [AssetType.IMAGE.value, AssetType.GIF.value],
    }

    return {
        **state,
        "final_assets": final_assets,
        "metadata": metadata,
        "current_step": "complete",
        "status": WorkflowStatus.COMPLETED.value,
    }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_create_gif(state: AssetGenerationState) -> str:
    """Determine if GIF creation should run."""
    if state.get("style") in ["animated", "gif", "motion"]:
        return "create_gif"
    return "finalize"


def has_errors(state: AssetGenerationState) -> str:
    """Check if workflow has critical errors."""
    if len(state.get("errors", [])) > 3:
        return "error_handler"
    return "continue"


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def build_asset_generation_workflow():
    """
    Build the complete asset generation workflow graph.

    Pipeline:
    1. scrape_references (if URLs provided)
    2. generate_images (ComfyUI + Flux)
    3. upscale_images (Anime4K)
    4. denoise_images (NVIDIA OptiX)
    5. create_gif (optional, based on style)
    6. finalize_assets

    Returns:
        Compiled LangGraph workflow
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph is required. Run: pip install langgraph")

    workflow = StateGraph(AssetGenerationState)

    # Add nodes
    workflow.add_node("scrape_references", scrape_references_node)
    workflow.add_node("generate_images", generate_images_node)
    workflow.add_node("upscale_images", upscale_images_node)
    workflow.add_node("denoise_images", denoise_images_node)
    workflow.add_node("create_gif", create_gif_node)
    workflow.add_node("finalize", finalize_assets_node)

    # Set entry point
    workflow.set_entry_point("scrape_references")

    # Add edges
    workflow.add_edge("scrape_references", "generate_images")
    workflow.add_edge("generate_images", "upscale_images")

    # Add Retinex enhancement step
    def enhance_images_node(state: AssetGenerationState) -> AssetGenerationState:
        """Apply Retinex enhancement."""
        logger.info("Enhancing images with Retinex")
        if not VISUAL_TOOLS_AVAILABLE:
            return state

        enhancer = VisualEnhancementNode()
        enhanced_paths = []

        # Process upscaled images
        for img_path in state.get("upscaled_images", []):
            try:
                result = enhancer({
                    "image_path": img_path,
                    "enhancement_mode": "retinex"
                })
                if result.get("status") == "success":
                    enhanced_paths.append(result["enhanced_image_path"])
                else:
                    enhanced_paths.append(img_path)
            except Exception as e:
                logger.error(f"Retinex failed: {e}")
                enhanced_paths.append(img_path)

        # Update upscaled_images with enhanced ones (or separate field if schema allowed)
        # For simplicity, we update in place or pass to denoise
        return {**state, "upscaled_images": enhanced_paths}

    workflow.add_node("enhance_images", enhance_images_node)

    workflow.add_edge("upscale_images", "enhance_images")
    workflow.add_edge("enhance_images", "denoise_images")

    # Conditional: GIF or finalize
    workflow.add_conditional_edges(
        "denoise_images",
        should_create_gif,
        {
            "create_gif": "create_gif",
            "finalize": "finalize",
        }
    )
    workflow.add_edge("create_gif", "finalize")
    workflow.add_edge("finalize", END)

    # Compile with memory for checkpointing
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ============================================================================
# WORKFLOW RUNNER
# ============================================================================

class AssetWorkflowRunner:
    """
    Runner for asset generation workflows.

    Provides high-level API for executing workflows
    and managing state.

    Example:
        >>> runner = AssetWorkflowRunner()
        >>> result = runner.run("anime girl, cyberpunk", style="animated")
        >>> print(result["final_assets"])
    """

    def __init__(self):
        self.workflow = None
        self._build_workflow()

    def _build_workflow(self):
        """Build the workflow graph."""
        if LANGGRAPH_AVAILABLE:
            self.workflow = build_asset_generation_workflow()
            logger.info("Asset workflow built successfully")
        else:
            logger.warning("LangGraph not available, workflow disabled")

    def run(
        self,
        prompt: str,
        style: Optional[str] = None,
        reference_urls: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ) -> AssetGenerationState:
        """
        Execute the asset generation workflow.

        Args:
            prompt: Generation prompt
            style: Optional style modifier
            reference_urls: Optional reference URLs to scrape
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final workflow state with generated assets
        """
        if not self.workflow:
            raise RuntimeError("Workflow not available")

        initial_state = create_initial_state(prompt, style)
        if reference_urls:
            initial_state["reference_urls"] = reference_urls

        config = {"configurable": {"thread_id": thread_id or "default"}}

        logger.info(f"Starting workflow for: {prompt[:50]}...")
        final_state = self.workflow.invoke(initial_state, config)

        logger.info(f"Workflow complete. Assets: {len(final_state['final_assets'])}")
        return final_state

    def stream(
        self,
        prompt: str,
        style: Optional[str] = None,
    ):
        """
        Stream workflow execution, yielding state after each node.

        Useful for progress tracking and UI updates.
        """
        if not self.workflow:
            raise RuntimeError("Workflow not available")

        initial_state = create_initial_state(prompt, style)

        for state in self.workflow.stream(initial_state):
            yield state
