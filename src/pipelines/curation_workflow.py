"""
Curation Workflow for Smart Asset Factory.
Implements the Generation -> Soft Scoring -> Interim Storage loop.
"""
import logging
from typing import Any, Dict, List, TypedDict


class CurationContent(TypedDict):
    path: str
    score: float
    metadata: Dict[str, Any]

class CurationState(TypedDict):
    """State for the curation loop."""
    # Input
    prompt: str
    style: str
    target_quota: int

    # Batch State
    current_batch_images: List[str]
    current_batch_scores: List[Dict[str, Any]]

    # Accumulators
    total_generated: int
    total_kept: int
    kept_assets: List[str] # Paths of kept assets

    # Control
    status: str
    errors: List[str]

# ============================================================================
# NODES
# ============================================================================

def generate_batch_node(state: CurationState) -> CurationState:
    """Generate a batch of 4 images."""
    logger.info(f"Generating batch. Total kept so far: {state['total_kept']}")

    # Default to ComfyUI, could be configurable
    generator = ComfyUINode()
    if not generator.validate():
        # Fallback for testing or if ComfyUI missing
        logger.warning("Generators not available, producing dummy data")
        dummy_path = "output/dummy_gen.png"
        return {**state, "current_batch_images": [dummy_path] * 4}

    results = []
    # Generate 4 images (logic might need adjustment depending on how generator handles batches)
    # ComfyUI often handles batches natively, but here we might loop or use 'batch_size' in input
    try:
        # Assuming generator supports batch_size or we loop.
        # For simplicity in this 'farm' context, let's assume single request = batch of 4 if configured
        # OR we just loop 4 times. Looping is safer for now.
        for _ in range(4):
            res = generator.execute({
                "prompt": f"{state['prompt']}, {state['style']}",
                "width": 512, # User requested 512x512
                "height": 512
            })
            if res.success and res.data.get("images"):
                results.extend(res.data["images"])

    except Exception as e:
        state["errors"].append(f"Generation failed: {e}")

    return {
        **state,
        "current_batch_images": results,
        "total_generated": state["total_generated"] + len(results)
    }

def score_batch_node(state: CurationState) -> CurationState:
    """Score the current batch using ViT."""
    scorer = ViTScoreNode()

    if not state["current_batch_images"]:
        return state

    res = scorer.execute({
        "image_paths": state["current_batch_images"],
        "prompt": state["style"], # Score against style expectation
        "threshold": 92.0 # User requirement: "If Point + 92 is keep"
    })

    if res.success:
        return {**state, "current_batch_scores": res.data["scores"]}

    state["errors"].append(f"Scoring failed: {res.error}")
    return state

def filter_and_store_node(state: CurationState) -> CurationState:
    """Filter kept images and store to Interim DB."""
    storage = ChromaStorageNode()

    new_kept = []

    for item in state.get("current_batch_scores", []):
        if item.get("qualified"):
            path = item["path"]
            score = item["score"]

            # Store to Interim Layer
            storage.execute({
                "operation": "add",
                "layer": "interim",
                "image_path": path,
                "metadata": {
                    "prompt": state["prompt"],
                    "style": state["style"],
                    "score": score,
                    "status": "waiting_human_review"
                }
            })
            new_kept.append(path)

    return {
        **state,
        "kept_assets": state["kept_assets"] + new_kept,
        "total_kept": state["total_kept"] + len(new_kept)
    }

def check_quota_node(state: CurationState) -> CurationState:
    """Check if we reached the target quota."""
    if state["total_kept"] >= state["target_quota"]:
        return {**state, "status": "complete"}
    return {**state, "status": "running"}

# ============================================================================
# GRAPH
# ============================================================================

def should_continue(state: CurationState) -> str:
    """Decide whether to loop or end."""
    if state["status"] == "complete":
        return END
    # Safety break
    if len(state["errors"]) > 10:
        logger.error("Too many errors, aborting curation loop")
        return END
    return "generate_batch"

def build_curation_workflow():
    """Build the LangGraph workflow."""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph required")

    workflow = StateGraph(CurationState)

    workflow.add_node("generate_batch", generate_batch_node)
    workflow.add_node("score_batch", score_batch_node)
    workflow.add_node("filter_store", filter_and_store_node)
    workflow.add_node("check_quota", check_quota_node)

    workflow.set_entry_point("generate_batch")

    workflow.add_edge("generate_batch", "score_batch")
    workflow.add_edge("score_batch", "filter_store")
    workflow.add_edge("filter_store", "check_quota")

    workflow.add_conditional_edges(
        "check_quota",
        should_continue
    )

    return workflow.compile()
