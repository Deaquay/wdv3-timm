# wdv3-timm

Image tagging using [SmilingWolf's WD Tagger V3](https://huggingface.co/SmilingWolf) models, loaded via [timm](https://github.com/huggingface/pytorch-image-models). Models are downloaded automatically from HuggingFace on first run. Supports local files, directories (batch), and URLs.

## Setup

1. Clone and enter the repo:
```sh
git clone https://github.com/Deaquay/wdv3-timm.git
cd wdv3-timm
```

2. Install [uv](https://docs.astral.sh/uv/) (if you don't have it):
```sh
./install-uv.sh        # Linux/macOS
install-uv.bat         # Windows
```

3. Run the installer:
```sh
./install.sh           # Linux/macOS
install.bat            # Windows
```

This creates a Python 3.13 venv and installs torch with `--torch-backend auto` (auto-detects your CUDA version). If uv isn't installed it falls back to pip.

4. Activate the venv:
```sh
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows
```

<details>
<summary>Manual setup (without uv)</summary>

```sh
python3 -m venv .venv
source .venv/bin/activate

# Install torch first — see https://pytorch.org/ for your platform
pip install torch torchvision

pip install -r requirements.txt
```
</details>

## Usage

```sh
# Single image
python wdv3_timm.py image.png

# Batch process a directory
python wdv3_timm.py /path/to/images/
python wdv3_timm.py -r /path/to/images/            # recursive into subdirs

# Tag from a URL (direct image link, max 20MB)
python wdv3_timm.py https://example.com/image.png
```

### Options

| Flag | Description | Default |
|---|---|---|
| `-m MODEL` | Model: `eva02`, `vit-large`, `vit`, `convnext`, `swinv2` | `eva02` |
| `-g THRESH` | General tag confidence threshold | `0.35` |
| `-c THRESH` | Character tag confidence threshold | `0.75` |
| `-b SIZE` | Batch size for directory processing | `8` |
| `-s SUFFIX` | Output as `filename_SUFFIX.txt` instead of `filename.txt` | none |
| `-q` | Quiet mode — tags to stdout only, no file output | off |
| `-f` | Force reprocess even if tag file exists | off |
| `-r` | Recurse into subdirectories | off |
| `--csv` | Output as CSV (`tag,confidence` per line) | off |

### Example

```
$ python wdv3_timm.py image.png

Using tag file format: filename.txt
Using cuda
Loading model 'eva02' from 'SmilingWolf/wd-eva02-large-tagger-v3'...
Loading tag list...
Creating data transform...
Processing single image...
--------
Caption: ganyu_(genshin_impact), 1girl, horns, solo, bell, ...
--------
Tags saved to: image.txt
--------
Ratings:
  general: 0.827
  sensitive: 0.199
  questionable: 0.001
  explicit: 0.001
Done!
```
