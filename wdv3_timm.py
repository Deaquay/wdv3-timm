import argparse
import csv
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request

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

MAX_URL_BYTES = 20 * 1024 * 1024  # 20 MB
VALID_URL_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif"}


@dataclass
class LabelData:
    names: list[str]
    rating: list[int]
    general: list[int]
    character: list[int]


@dataclass
class ScriptOptions:
    path: str
    gen_threshold: float = 0.35
    char_threshold: float = 0.70
    batch_size: int = 8
    quiet: bool = False
    force: bool = False
    recursive: bool = False
    suffix: str | None = None
    model: str = "eva02"
    csv: bool = False


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    px = max(image.size)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def prepare_image(image_path: Path) -> Image.Image:
    img = Image.open(image_path)
    img = pil_ensure_rgb(img)
    return pil_pad_square(img)


def get_tags_path(image_path: Path, suffix: str | None, use_csv: bool = False) -> Path:
    ext = ".csv" if use_csv else ".txt"
    if suffix:
        return image_path.parent / f"{image_path.stem}_{suffix}{ext}"
    return image_path.with_suffix(ext)


def load_labels_hf(
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    names = []
    rating = []
    general = []
    character = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            names.append(row["name"])
            category = int(row["category"])
            if category == 9:
                rating.append(i)
            elif category == 0:
                general.append(i)
            elif category == 4:
                character.append(i)

    return LabelData(names=names, rating=rating, general=general, character=character)


def get_tags(probs, labels: LabelData, gen_threshold: float, char_threshold: float):
    pairs = list(zip(labels.names, probs))

    rating_labels = {pairs[i][0]: pairs[i][1] for i in labels.rating}

    gen_labels = {pairs[i][0]: pairs[i][1] for i in labels.general if pairs[i][1] > gen_threshold}
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    char_labels = {pairs[i][0]: pairs[i][1] for i in labels.character if pairs[i][1] > char_threshold}
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    combined_names = list(char_labels) + list(gen_labels)
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\\(").replace(")", "\\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


def should_process_image(image_path: Path, suffix: str | None, force: bool = False, use_csv: bool = False) -> bool:
    return force or not get_tags_path(image_path, suffix, use_csv).exists()


def collect_image_files(path: Path, recursive: bool = False) -> list[Path]:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = path.rglob("*.*") if recursive else path.glob("*.*")
    return [f for f in files if f.suffix.lower() in valid_extensions]


def write_output(tags_path: Path, taglist: str, char_labels: dict, gen_labels: dict, use_csv: bool):
    with open(tags_path, "w", encoding="utf-8") as f:
        if use_csv:
            for tag, score in char_labels.items():
                f.write(f"{tag},{score:.4f}\n")
            for tag, score in gen_labels.items():
                f.write(f"{tag},{score:.4f}\n")
        else:
            f.write(taglist)


def process_single_image(image_path: Path, model, labels: LabelData,
                         opts: ScriptOptions, transform, device) -> bool:
    import torch
    try:
        img_input = prepare_image(image_path)
        inputs = transform(img_input).unsqueeze(0)
        inputs = inputs[:, [2, 1, 0]]  # RGB to BGR

        with torch.inference_mode():
            outputs = model(inputs.to(device)).sigmoid().cpu()

        caption, taglist, ratings, character, general = get_tags(
            probs=outputs.squeeze(0).tolist(),
            labels=labels,
            gen_threshold=opts.gen_threshold,
            char_threshold=opts.char_threshold,
        )

        if opts.quiet:
            if opts.csv:
                for tag, score in character.items():
                    print(f"{tag},{score:.4f}")
                for tag, score in general.items():
                    print(f"{tag},{score:.4f}")
            else:
                print(taglist)
        else:
            tags_path = get_tags_path(image_path, opts.suffix, opts.csv)
            write_output(tags_path, taglist, character, general, opts.csv)

            print("--------")
            print(f"Caption: {caption}")
            print("--------")
            print(f"Tags saved to: {tags_path}")
            print("--------")
            print("Ratings:")
            for k, v in ratings.items():
                print(f"  {k}: {v:.3f}")

        return True
    except Exception as e:
        if not opts.quiet:
            print(f"Error processing {image_path.name}: {e}")
        return False


def process_batch(image_paths: list[Path], should_process_flags: list[bool],
                  model, labels: LabelData, opts: ScriptOptions, transform, device, pbar):
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
            outputs = model(inputs.to(device)).sigmoid().cpu()

        for i, image_path in enumerate(valid_paths):
            caption, taglist, _, character, general = get_tags(
                probs=outputs[i].tolist(),
                labels=labels,
                gen_threshold=opts.gen_threshold,
                char_threshold=opts.char_threshold,
            )

            tags_path = get_tags_path(image_path, opts.suffix, opts.csv)
            write_output(tags_path, taglist, character, general, opts.csv)
            pbar.update(1)


def handle_directory(path: Path, opts: ScriptOptions, model, labels, transform, device):
    print("Collecting image files...")
    image_files = collect_image_files(path, opts.recursive)

    should_process_flags = [should_process_image(img, opts.suffix, opts.force, opts.csv) for img in image_files]

    to_process_count = sum(should_process_flags)
    total_count = len(image_files)
    skipped_count = total_count - to_process_count

    if not opts.quiet:
        print(f"Found {total_count} images, {skipped_count} already tagged, processing {to_process_count}")

    if to_process_count == 0:
        if opts.quiet:
            for img_path in image_files:
                tags_path = get_tags_path(img_path, opts.suffix, opts.csv)
                if tags_path.exists():
                    print(tags_path.read_text(encoding="utf-8"))
        else:
            print("No new images to process. Done!")
        return

    pbar = tqdm(total=total_count, unit="image", unit_scale=True)

    for i in range(0, len(image_files), opts.batch_size):
        batch_images = image_files[i:i + opts.batch_size]
        batch_flags = should_process_flags[i:i + opts.batch_size]
        process_batch(batch_images, batch_flags, model, labels, opts, transform, device, pbar)

    pbar.close()
    print("Done!")


def handle_single_file(file_path: Path, opts: ScriptOptions, model, labels, transform, device):
    tags_path = get_tags_path(file_path, opts.suffix, opts.csv)

    if not should_process_image(file_path, opts.suffix, opts.force, opts.csv):
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


def validate_image_url(url: str):
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext not in VALID_URL_EXTENSIONS:
        raise ValueError(
            f"URL must point directly to an image file ({', '.join(VALID_URL_EXTENSIONS)}), got '{ext or 'none'}'"
        )


def handle_url(url: str, opts: ScriptOptions, model, labels, transform, device):
    import torch

    validate_image_url(url)

    if not opts.quiet:
        print("Fetching image from URL...")

    request = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; wdv3-timm image tagger)",
    })
    response = urlopen(request)

    content_length = response.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_URL_BYTES:
        raise ValueError(f"Image too large ({int(content_length) // 1024 // 1024}MB), max is 20MB")

    data = response.read(MAX_URL_BYTES + 1)
    if len(data) > MAX_URL_BYTES:
        raise ValueError("Image too large (>20MB), max is 20MB")

    img = Image.open(BytesIO(data))
    img = pil_ensure_rgb(img)
    img = pil_pad_square(img)

    inputs = transform(img).unsqueeze(0)
    inputs = inputs[:, [2, 1, 0]]  # RGB to BGR

    with torch.inference_mode():
        outputs = model(inputs.to(device)).sigmoid().cpu()

    caption, taglist, ratings, character, general = get_tags(
        probs=outputs.squeeze(0).tolist(),
        labels=labels,
        gen_threshold=opts.gen_threshold,
        char_threshold=opts.char_threshold,
    )

    if opts.csv:
        for tag, score in character.items():
            print(f"{tag},{score:.4f}")
        for tag, score in general.items():
            print(f"{tag},{score:.4f}")
    else:
        print(taglist)

    if not opts.quiet:
        print("--------")
        print(f"Caption: {caption}")
        print("--------")
        print("Ratings:")
        for k, v in ratings.items():
            print(f"  {k}: {v:.3f}")


def parse_args() -> ScriptOptions:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("path", type=str, help="Image file, directory, or URL")
    parser.add_argument("-g", "--gen-threshold", type=float, default=0.35,
                        help="General tag threshold (default: 0.35)")
    parser.add_argument("-c", "--char-threshold", type=float, default=0.75,
                        help="Character tag threshold (default: 0.75)")
    parser.add_argument("-b", "--batch-size", type=int, default=8,
                        help="Batch size for directory processing (default: 8)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Run in quiet mode, output only tags, single image only.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Process images even if they already have tag files")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Recursively search subdirectories for images")
    parser.add_argument("-s", "--suffix", type=str, default=None,
                        help="Suffix for tag files (e.g. -s tags -> filename_tags.txt)")
    parser.add_argument("-m", "--model", type=str, default="eva02",
                        choices=list(MODEL_REPO_MAP.keys()),
                        help="Model to use (default: eva02)")
    parser.add_argument("--csv", action="store_true",
                        help="Save output in CSV format (tag,confidence) per line")

    args = parser.parse_args()

    return ScriptOptions(
        path=args.path,
        gen_threshold=args.gen_threshold,
        char_threshold=args.char_threshold,
        batch_size=args.batch_size,
        quiet=args.quiet,
        force=args.force,
        recursive=args.recursive,
        suffix=args.suffix,
        model=args.model,
        csv=args.csv,
    )


def main(opts: ScriptOptions):
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")

    repo_id = MODEL_REPO_MAP[opts.model]
    is_url = opts.path.startswith(("http://", "https://"))

    if not is_url:
        path = Path(opts.path).resolve()
        is_file = path.is_file()
        is_dir = path.is_dir()

        if not (is_file or is_dir):
            raise FileNotFoundError(f"Path not found: {path}")

        ext_desc = ".csv" if opts.csv else ".txt"
        suffix_desc = f"filename_{opts.suffix}{ext_desc}" if opts.suffix else f"filename{ext_desc}"

        if not opts.quiet:
            print(f"Using tag file format: {suffix_desc}")
            if opts.csv:
                print("Output mode: CSV (tag,confidence)")

        # Check for existing work before loading heavy models
        if is_file and not should_process_image(path, opts.suffix, opts.force, opts.csv):
            if opts.quiet:
                tags_path = get_tags_path(path, opts.suffix, opts.csv)
                if tags_path.exists():
                    print(tags_path.read_text(encoding="utf-8"))
            else:
                print(f"Skipping {path.name}: already tagged.")
            return

        if is_dir:
            image_files = collect_image_files(path, opts.recursive)
            if not any(should_process_image(img, opts.suffix, opts.force, opts.csv) for img in image_files):
                if opts.quiet:
                    for img_path in image_files:
                        tags_path = get_tags_path(img_path, opts.suffix, opts.csv)
                        if tags_path.exists():
                            print(tags_path.read_text(encoding="utf-8"))
                else:
                    print(f"Found {len(image_files)} images, all already tagged. Skipping model load.")
                return

    import timm
    from timm.data import create_transform, resolve_data_config
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not opts.quiet:
        print(f"Using {device.type}")
        print(f"Loading model '{opts.model}' from '{repo_id}'...")

    model = timm.create_model("hf-hub:" + repo_id, pretrained=True).eval().to(device)

    if not opts.quiet:
        print("Loading tag list...")
    labels = load_labels_hf(repo_id=repo_id)

    if not opts.quiet:
        print("Creating data transform...")
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    if is_url:
        handle_url(opts.path, opts, model, labels, transform, device)
    elif is_file:
        handle_single_file(path, opts, model, labels, transform, device)
    else:
        handle_directory(path, opts, model, labels, transform, device)


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
