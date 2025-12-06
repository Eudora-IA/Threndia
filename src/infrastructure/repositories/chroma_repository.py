        return self._stores[layer]

    def validate(self) -> bool:
        """Check if ChromaDB is available (implied by import success of core)."""
        return True

    @log_execution
    def execute(self, input_data: Dict[str, Any]) -> NodeResult:
        """
        Execute storage operation.

        Args:
            input_data:
                - operation: "add" or "search"
                - layer: "interim" or "gold" (default "interim")
                - image_path: str (for add)
                - metadata: dict (for add)
                - query: str (for search)
                - limit: int
        """
        try:
            operation = input_data.get("operation", "add")
            layer = input_data.get("layer", "interim")
            store = self._get_store(layer)

            if operation == "add":
                image_path = input_data.get("image_path")
                if not image_path:
                    return NodeResult(success=False, error="Image path required for add")

                metadata = input_data.get("metadata", {})
                doc_id = store.add_image(image_path, metadata=metadata)

                return NodeResult(
                    success=True,
                    data={"doc_id": doc_id, "layer": layer},
                    metadata={"operation": "add"}
                )

            elif operation == "search":
                query = input_data.get("query")
                limit = input_data.get("limit", 10)

                if not query:
                     return NodeResult(success=False, error="Query required for search")

                results = store.search_images_by_text(query, limit=limit)
                return NodeResult(
                    success=True,
                    data={"results": results},
                    metadata={"operation": "search", "count": len(results)}
                )

            else:
                return NodeResult(success=False, error=f"Unknown operation: {operation}")

        except Exception as e:
            logger.error(f"Storage operation failed: {e}")
            return NodeResult(success=False, error=str(e))
