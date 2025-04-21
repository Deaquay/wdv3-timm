from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import threading
import queue
import argparse

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from tqdm import tqdm

MODEL_REPO_MAP = {
    "eva02": "SmilingWolf/wd-eva02-large-tagger-v3",
    "vit-large": "SmilingWolf/wd-vit-large-tagger-v3",
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
}

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


@dataclass
class ScriptOptions:
    path: Path
    gen_threshold: float = 0.35
    char_threshold: float = 0.75 
    batch_size: int = 8
    quiet: bool = False
    force: bool = False
    recursive: bool = False
    suffix: str = "tags"
    model: str = "eva02"


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert image to RGB format."""
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    """Pad image to square with white background."""
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def prepare_image(image_path: Path) -> Image.Image:
    """Load and prepare an image for model input."""
    img = Image.open(image_path)
    img = pil_ensure_rgb(img)
    img = pil_pad_square(img)
    return img.convert("RGB")


def get_tags_path(image_path: Path, suffix: str) -> Path:
    """Get the path for the tags text file based on the suffix option."""
    if suffix.lower() == "none":
        # Same name as image, but with .txt extension
        return image_path.with_suffix(".txt")
    else:
        # Add the suffix before .txt
        return image_path.parent / f"{image_path.stem}_{suffix}.txt"


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    """Load labels from HuggingFace repository."""
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
    probs,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
) -> Tuple[str, str, Dict, Dict, Dict]:
    """Convert model output probabilities to tags."""
    # Convert indices+probs to labels
    probs = list(zip(labels.names, probs))

    # First 4 labels are actually ratings
    rating_labels = dict([probs[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\(").replace(")", "\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


def should_process_image(image_path: Path, suffix: str, force: bool = False) -> bool:
    """Check if an image should be processed based on if it already has a tags file."""
    tags_path = get_tags_path(image_path, suffix)
    return force or not tags_path.exists()


def collect_image_files(path: Path, recursive: bool = False) -> List[Path]:
    """Collect all image files from the specified path."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    if recursive:
        files = path.rglob("*.*")
    else:
        files = path.glob("*.*")
    
    return [f for f in files if f.suffix.lower() in valid_extensions]


def process_single_image(image_path: Path, model, labels: LabelData, 
                        opts: ScriptOptions, transform, device) -> bool:
    """Process a single image and save the tags to a .txt file."""
    import torch
    try:
        img_input = prepare_image(image_path)
        inputs = transform(img_input).unsqueeze(0)
        inputs = inputs[:, [2, 1, 0]]  # RGB to BGR

        with torch.inference_mode():
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.nn.functional.sigmoid(outputs)
            outputs = outputs.cpu()

        caption, taglist, ratings, character, general = get_tags(
            probs=outputs.squeeze(0).numpy(),
            labels=labels,
            gen_threshold=opts.gen_threshold,
            char_threshold=opts.char_threshold,
        )

        if opts.quiet:
            # In quiet mode, only print the taglist and skip writing the file
            print(taglist)
        else:
            # Write the taglist to the tags file
            tags_path = get_tags_path(image_path, opts.suffix)
            with open(tags_path, "w", encoding="utf-8") as f:
                f.write(taglist)

            # Print detailed information
            print("--------")
            print(f"Caption: {caption}")
            print("--------")
            print(f"Tags: {taglist}")
            print(f"Tags saved to: {tags_path}")

            print("--------")
            print("Ratings:")
            for k, v in ratings.items():
                print(f"  {k}: {v:.3f}")

            print("--------")
            print(f"Character tags (threshold={opts.char_threshold}):")
            for k, v in character.items():
                print(f"  {k}: {v:.3f}")

            print("--------")
            print(f"General tags (threshold={opts.gen_threshold}):")
            for k, v in general.items():
                print(f"  {k}: {v:.3f}")

        return True
    except Exception as e:
        if not opts.quiet:
            print(f"Error processing {image_path.name}: {e}")
        return False


def process_batch(image_paths: List[Path], should_process_flags: List[bool], 
                model, labels: LabelData, opts: ScriptOptions, transform, device, pbar):
    """Process a batch of images with the model."""
    import torch
    
    inputs = []
    valid_paths = []

    for i, image_path in enumerate(image_paths):
        if not should_process_flags[i]:
            pbar.update(1)
            continue

        try:
            img_input = prepare_image(image_path)
            inputs.append(transform(img_input).unsqueeze(0))
            valid_paths.append(image_path)
        except Exception as e:
            if not opts.quiet:
                print(f"Error processing {image_path.name}: {e}")
            pbar.update(1)

    if inputs:
        inputs = torch.cat(inputs)
        inputs = inputs[:, [2, 1, 0]]  # RGB to BGR

        with torch.inference_mode():
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.nn.functional.sigmoid(outputs)
            outputs = outputs.cpu()

        for i, image_path in enumerate(valid_paths):
            caption, taglist, _, _, _ = get_tags(
                probs=outputs[i].squeeze(0).numpy(),
                labels=labels,
                gen_threshold=opts.gen_threshold,
                char_threshold=opts.char_threshold,
            )

            tags_path = get_tags_path(image_path, opts.suffix)
            with open(tags_path, "w", encoding="utf-8") as f:
                f.write(taglist)

            pbar.update(1)


def worker(queue_data, model, labels, opts, transform, device, pbar):
    """Worker thread for batch processing of images."""
    while True:
        item = queue_data.get()
        if item is None:
            break

        image_paths, should_process_flags = item
        process_batch(image_paths, should_process_flags, model, labels, opts, transform, device, pbar)
        queue_data.task_done()


def handle_directory(path: Path, opts: ScriptOptions, model, labels, transform, device):
    """Process all images in a directory."""
    # Collect and filter image files before processing
    print("Collecting image files...")
    image_files = collect_image_files(path, opts.recursive)
    
    # Check which files need processing
    should_process_flags = [should_process_image(img_path, opts.suffix, opts.force) for img_path in image_files]
    
    # Count how many files need processing
    to_process_count = sum(should_process_flags)
    total_count = len(image_files)
    skipped_count = total_count - to_process_count
    
    if not opts.quiet:
        print(f"Found {total_count} images, {skipped_count} already tagged, processing {to_process_count}")
    
    # If no files to process, exit early
    if to_process_count == 0:
        if opts.quiet:
            # In quiet mode, output existing tags
            for img_path in image_files:
                tags_path = get_tags_path(img_path, opts.suffix)
                if tags_path.exists():
                    print(tags_path.read_text(encoding="utf-8"))
        else:
            print("No new images to process. Done!")
        return
    
    # Initialize progress bar
    pbar = tqdm(total=total_count, unit="image", unit_scale=True)

    # Start worker threads
    image_queue = queue.Queue()
    threads = []
    num_threads = min(opts.batch_size, to_process_count)
    
    for _ in range(num_threads):
        thread = threading.Thread(target=worker, 
                                args=(image_queue, model, labels, opts, transform, device, pbar))
        thread.daemon = True
        thread.start()
        threads.append(thread)

    # Queue up batches of images
    batch_size = opts.batch_size
    for i in range(0, len(image_files), batch_size):
        batch_images = image_files[i:i+batch_size]
        batch_flags = should_process_flags[i:i+batch_size]
        image_queue.put((batch_images, batch_flags))

    # Wait for all tasks to complete
    image_queue.join()

    # Stop worker threads
    for _ in threads:
        image_queue.put(None)
    for thread in threads:
        thread.join()

    pbar.close()
    print("Done!")


def handle_single_file(file_path: Path, opts: ScriptOptions, model, labels, transform, device):
    """Process a single image file."""
    tags_path = get_tags_path(file_path, opts.suffix)
    
    # If skipping, but in quiet mode, just output contents of tags file
    if not should_process_image(file_path, opts.suffix, opts.force):
        if opts.quiet:
            if tags_path.exists():
                print(tags_path.read_text(encoding="utf-8"))
            else:
                print(f"[Missing tags file for {file_path.name}]")
        else:
            print(f"Skipping {file_path.name}: tags file already exists at {tags_path}")
        return
    
    if not opts.quiet:
        print("Processing single image...")
    
    process_single_image(file_path, model, labels, opts, transform, device)
    
    if not opts.quiet:
        print("Done!")


def parse_args() -> ScriptOptions:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Add positional argument
    parser.add_argument("path", type=Path, help="Path to image file or directory")
    
    # Add optional arguments with cleaner help text
    parser.add_argument("-g", "--gen-threshold", type=float, default=0.35, 
                      help="General tag threshold (default: 0.35)")
    parser.add_argument("-c", "--char-threshold", type=float, default=0.75, 
                      help="Character tag threshold (default: 0.75)")
    parser.add_argument("-b", "--batch-size", type=int, default=8, 
                      help="Number of images to process at once (default: 8)")
    parser.add_argument("-q", "--quiet", action="store_true", 
                      help="Run in quiet mode, output only tags, single image only.")
    parser.add_argument("-f", "--force", action="store_true", 
                      help="Process images even if they already have txt files")
    parser.add_argument("-r", "--recursive", action="store_true", 
                      help="Recursively search subdirectories for images")
    parser.add_argument("-s", "--suffix", type=str, default="tags", 
                      help="Suffix for tag files: 'tags' for filename_tags.txt, 'none' for filename.txt (default: tags)")
    parser.add_argument("-m", "--model", type=str, default="eva02", 
                      choices=list(MODEL_REPO_MAP.keys()),
                      help=f"Model to use (default: eva02)")
    
    args = parser.parse_args()
    
    # Convert to dataclass
    return ScriptOptions(
        path=args.path,
        gen_threshold=args.gen_threshold,
        char_threshold=args.char_threshold,
        batch_size=args.batch_size,
        quiet=args.quiet,
        force=args.force,
        recursive=args.recursive,
        suffix=args.suffix,
        model=args.model
    )


def main(opts: ScriptOptions):
    # Validate model selection
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
        
    # Get the tag file suffix format description
    if opts.suffix.lower() == "none":
        suffix_desc = "same name as image (.txt)"
    else:
        suffix_desc = f"filename_{opts.suffix}.txt"

    repo_id = MODEL_REPO_MAP.get(opts.model)
    path = Path(opts.path).resolve()

    # Check if path is a file or directory
    is_file = path.is_file()
    is_dir = path.is_dir()

    if not (is_file or is_dir):
        raise FileNotFoundError(f"Path not found: {path}")

    if not opts.quiet:
        print(f"Using tag file format: {suffix_desc}")

    # Early exit for single files with existing tags file (avoid loading model)
    if is_file and not should_process_image(path, opts.suffix, opts.force):
        if opts.quiet:
            tags_path = get_tags_path(path, opts.suffix)
            if tags_path.exists():
                print(tags_path.read_text(encoding="utf-8"))
            else:
                print(f"[Missing tags file for {path.name}]")
        else:
            print(f"Skipping {path.name}: tags file already exists at {get_tags_path(path, opts.suffix)}")
        return
    
    # Early exit for directories with no files to process (avoid loading model)
    if is_dir:
        # Quick scan to see if there's anything to process
        image_files = collect_image_files(path, opts.recursive)
        should_process_any = any(should_process_image(img, opts.suffix, opts.force) for img in image_files)
        
        if not should_process_any:
            if opts.quiet:
                # Output existing tags in quiet mode
                for img_path in image_files:
                    tags_path = get_tags_path(img_path, opts.suffix)
                    if tags_path.exists():
                        print(tags_path.read_text(encoding="utf-8"))
            else:
                print(f"Found {len(image_files)} images, all already tagged. Skipping model load.")
            return
    
    # Now we know we need to load the model
    import torch
    import timm
    from timm.data import create_transform, resolve_data_config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print startup info if not in quiet mode or if processing directory
    if not opts.quiet or is_dir:
        print(f"Using {device.type}")
        if device.type == "cpu":
            print("Warning: Running on CPU will be very slow compared to GPU.")
        print(f"Loading model '{opts.model}' from '{repo_id}'...")
    
    # Load the model
    model = timm.create_model("hf-hub:" + repo_id, pretrained=True)
    model.eval()
    state_dict = timm.models.load_state_dict_from_hf(repo_id)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Load tag list
    if not opts.quiet or is_dir:
        print("Loading tag list...")
    labels: LabelData = load_labels_hf(repo_id=repo_id)

    # Create data transform
    if not opts.quiet or is_dir:
        print("Creating data transform...")
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    # Process according to path type
    if is_file:
        handle_single_file(path, opts, model, labels, transform, device)
    else:
        handle_directory(path, opts, model, labels, transform, device)


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
