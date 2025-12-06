"""
    # We'll use a dummy query "" or "*" if supported, essentially "get recent".
    # Or strict metadata filter: {"status": "waiting_human_review"}

    res = storage.execute({
        "operation": "search",
        "layer": "interim", # Pull from Interim
        "query": "high quality", # Dummy query to get things
        "limit": limit
        # In a real app, we'd add strict metadata filtering here if the Node exposed it.
        # Assuming the Node/Store allows passing filters in 'metadata' or 'filters' key?
        # Checking ChromaStorageNode... it calls search_images_by_text(query, limit).
        # It doesn't strictly expose 'where' filter in the simple wrapper yet.
        # We might need to update ChromaStorageNode or just rely on similarity for now.
    })

    paths = []
    ids = []

    if res.success and res.data.get("results"):
        for item in res.data["results"]:
            # item structure from vector_database.py: {'id':..., 'content': path, ...}
            # Wait, content is path? Check vector_database.py add_image doc:
            # documents=[image_path]. Yes.

            # We filter for "waiting" status if possible,
            # but for MVP just show what we find.
            paths.append(item.get("content"))
            ids.append(item.get("id"))

    # Pad to 4 if needed
    while len(paths) < 4:
        paths.append(None)
        ids.append(None)

    return paths[:4], ids[:4]

def promote_image(doc_id, path):
    """Promote an image to Gold layer."""
    if not doc_id or not path:
        return "No image selected"

    logger.info(f"Promoting {doc_id} to Gold")

    # 1. Add to Gold
    res = storage.execute({
        "operation": "add",
        "layer": "gold",
        "image_path": path,
        "metadata": {"source": "interim_promotion", "original_id": doc_id}
    })

    # 2. Delete from Interim (Simulated by updating status if we could,
    # but strictly we might want to just move it.
    # For MVP, we just Copy to Gold. Cleanup of Interim is separate task.)

    if res.success:
        return f"Promoted {os.path.basename(path)} to Gold!"
    return f"Failed: {res.error}"

def refresh_view():
    paths, ids = fetch_candidates()
    return paths[0], paths[1], paths[2], paths[3], ids[0], ids[1], ids[2], ids[3], "Loaded new batch"

# UI Construction
with gr.Blocks(title="Fazenda Curation Station", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚜 Fazenda Curation Station")
    gr.Markdown("Review 'Interim' assets. Select the best to promote to the 'Gold' layer.")

    # State storage
    id_state_0 = gr.State()
    id_state_1 = gr.State()
    id_state_2 = gr.State()
    id_state_3 = gr.State()

    with gr.Row():
        with gr.Column():
            img0 = gr.Image(label="Option 1", type="filepath", interactive=False)
            btn0 = gr.Button("🏆 Select #1")
        with gr.Column():
            img1 = gr.Image(label="Option 2", type="filepath", interactive=False)
            btn1 = gr.Button("🏆 Select #2")

    with gr.Row():
        with gr.Column():
            img2 = gr.Image(label="Option 3", type="filepath", interactive=False)
            btn2 = gr.Button("🏆 Select #3")
        with gr.Column():
            img3 = gr.Image(label="Option 4", type="filepath", interactive=False)
            btn3 = gr.Button("🏆 Select #4")

    status_msg = gr.Textbox(label="Status", interactive=False)
    next_batch_btn = gr.Button("Next Batch (Reject All)", variant="secondary")

    # Actions
    def on_select_0(id_val, path_val):
        res = promote_image(id_val, path_val)
        p, i = fetch_candidates()
        return *p, *i, res

    def on_select_1(id_val, path_val):
        res = promote_image(id_val, path_val)
        p, i = fetch_candidates()
        return *p, *i, res

    def on_select_2(id_val, path_val):
        res = promote_image(id_val, path_val)
        p, i = fetch_candidates()
        return *p, *i, res

    def on_select_3(id_val, path_val):
        res = promote_image(id_val, path_val)
        p, i = fetch_candidates()
        return *p, *i, res

    def on_next():
        p, i = fetch_candidates()
        return *p, *i, "Skipped batch"

    # Wiring
    # Note: Gradio event handlers need to match return signature
    # 4 images, 4 states, 1 status
    outputs = [img0, img1, img2, img3, id_state_0, id_state_1, id_state_2, id_state_3, status_msg]

    btn0.click(on_select_0, inputs=[id_state_0, img0], outputs=outputs)
    btn1.click(on_select_1, inputs=[id_state_1, img1], outputs=outputs)
    btn2.click(on_select_2, inputs=[id_state_2, img2], outputs=outputs)
    btn3.click(on_select_3, inputs=[id_state_3, img3], outputs=outputs)

    next_batch_btn.click(on_next, outputs=outputs)

    # Load on start
    demo.load(refresh_view, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
