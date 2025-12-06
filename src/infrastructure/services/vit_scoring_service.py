        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers library not found. ViTScoreNode will fail.")

    def _load_model(self):
        """Lazy load the model."""
        if self.model is None and TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading CLIP model: {self.model_id} on {self.device}...")
                self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(self.model_id)
                logger.info("CLIP model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                raise

    def validate(self) -> bool:
        """Check if libraries are available."""
        return TRANSFORMERS_AVAILABLE

    @log_execution
    def execute(self, input_data: Dict[str, Any]) -> NodeResult:
        """
        Score an image or batch of images.

        Args:
            input_data:
                - image_paths: List[str] or str
                - prompt: str (Style/Quality prompt to compare against)
                - threshold: float (default 0.92 for pass/fail check in metadata)

        Returns:
            NodeResult with 'scores' list and 'qualified_paths'.
        """
        if not self.validate():
            return NodeResult(success=False, error="Transformers library missing")

        start_time = 0 # simple placeholder, decorator handles real timing

        try:
            self._load_model()

            image_paths = input_data.get("image_paths", [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            prompt = input_data.get("prompt", "high quality, masterpiece, detailed")
            threshold = float(input_data.get("threshold", 22.0)) # CLIP raw logits can be higher, roughly 20-30 range.
            # Note: Normalized cosine similarity is 0-1, but CLIPModel default call returns logits_per_image.
            # We will use probability or normalized score.

            # Load images
            images = []
            valid_paths = []
            for p in image_paths:
                try:
                    path_obj = Path(p)
                    if path_obj.exists():
                        images.append(Image.open(path_obj))
                        valid_paths.append(str(path_obj))
                except Exception as e:
                    logger.warning(f"Could not load image {p}: {e}")

            if not images:
                return NodeResult(success=False, error="No valid images to score")

            # Process
            inputs = self.processor(
                text=[prompt],
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # logits_per_image: [batch_size, text_batch_size]
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1) # This compares image against the TEXTS provided.
            # If we only have 1 text, softmax will be 1.0.
            # We need raw similarity or probability against a negative prompt to be meaningful
            # OR just use the raw logit / cosine similarity.

            # Let's use raw cosine similarity (normalized)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

            # similarity [batch_size, 1]
            similarity = (image_embeds @ text_embeds.t()).squeeze(1)

            scores = similarity.tolist() # range -1 to 1 usually

            # User requirement: "If Point (per art styyle) + 92 is keep"
            # Assuming threshold 0.22 ~ 0.25 is decent for raw cosine sim in CLIP.
            # If user means 92/100, that implies a very specific metric or training.
            # We will Map 0-1 to 0-100 for user friendliness.

            scaled_scores = [s * 100 for s in scores]

            qualified_paths = []
            results = []

            for path, score in zip(valid_paths, scaled_scores):
                is_qualified = score >= threshold
                if is_qualified:
                    qualified_paths.append(path)
                results.append({
                    "path": path,
                    "score": score,
                    "qualified": is_qualified
                })

            return NodeResult(
                success=True,
                data={
                    "scores": results,
                    "qualified_paths": qualified_paths,
                    "count": len(results),
                    "qualified_count": len(qualified_paths)
                },
                metadata={
                    "model": self.model_id,
                    "threshold": threshold,
                    "prompt": prompt
                }
            )

        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return NodeResult(success=False, error=str(e))
