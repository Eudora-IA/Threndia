from pathlib import Path

from PIL import Image, ImageDraw


def create_dummy_dataset(output_dir: str = "data/datasets/dummy", count: int = 5):
    """
    Creates a dummy dataset with placeholder images and captions for testing training pipelines.

    Args:
        output_dir: Directory to save the dataset.
        count: Number of image-caption pairs to generate.
    """
    root = Path(__file__).parent.parent
    dataset_path = root / output_dir
    dataset_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating dummy dataset in: {dataset_path}")

    colors = ["red", "green", "blue", "yellow", "purple"]

    for i in range(count):
        # 1. Create Image
        color = colors[i % len(colors)]
        img = Image.new('RGB', (1024, 1024), color=color)
        d = ImageDraw.Draw(img)
        d.text((400, 500), f"Dummy {i}", fill="white")

        img_name = f"image_{i}.jpg"
        img.save(dataset_path / img_name)

        # 2. Create Caption
        txt_name = f"image_{i}.txt"
        caption = f"dummy_concept style, a simple {color} square image number {i}"

        with open(dataset_path / txt_name, "w") as f:
            f.write(caption)

        print(f"  Generated {img_name} + {txt_name}")

    print("✅ Dummy dataset created successfully!")

if __name__ == "__main__":
    create_dummy_dataset()
