# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WDV3-timm is a Python utility for running WD Tagger V3 image tagging models via the `timm` (PyTorch Image Models) library. It analyzes images and generates descriptive tags across categories (general, character, rating). Supports single-file, batch directory, and URL input.

## Setup

```bash
# Install uv first (recommended)
./install-uv.sh                 # Linux/macOS
install-uv.bat                  # Windows

# Then install dependencies
./install.sh                    # Linux/macOS
install.bat                     # Windows

source .venv/bin/activate       # Activate venv
```

Install scripts use uv if available (creates Python 3.13 venv, installs torch with `--torch-backend auto` for correct CUDA detection). Falls back to pip if uv is not installed. Uses `timm` from GitHub main branch (not PyPI stable).

## Running

```bash
# Single image (default model: eva02)
python wdv3_timm.py path/to/image.png

# Batch process directory
python wdv3_timm.py /path/to/images/

# With options
python wdv3_timm.py -m vit -r -g 0.4 -c 0.8 -s mytags /path/to/images/
python wdv3_timm.py -q image.png          # stdout only, no file output
python wdv3_timm.py --csv image.png       # CSV format output
python wdv3_timm.py -f image.png          # force reprocess already-tagged

# Tag from URL (direct image link, max 20MB)
python wdv3_timm.py https://example.com/image.png
python wdv3_timm.py -q https://example.com/image.jpg
```

## Architecture

All logic is in `wdv3_timm.py` (single-file project).

**Available models** (HuggingFace repos mapped in `REPO_MAP`): `eva02` (default), `vit-large`, `vit`, `convnext`, `swinv2`.

**Processing flow:**
1. CLI args parsed via `argparse` into `ScriptOptions` dataclass
2. Input dispatches to `handle_url()`, `handle_single_file()`, or `handle_directory()`
3. Existing tag files checked before loading the model (avoids heavy model load if no work needed)
4. Images prepared: ensure RGB, pad to square, resize to model input size
5. Model inference produces probabilities; `get_tags()` applies thresholds and categorizes results
6. Output written as TXT (comma-separated tags) or CSV (tag,confidence)

**URL mode** validates extension (png/jpg/jpeg/webp/avif) and enforces a 20MB size limit. Output always goes to stdout.

**Batch processing** iterates sequentially over batches (default size 8) with tqdm progress tracking. GPU handles parallelism via batched inference.

**Key dataclasses:** `LabelData` (tag names + category indices), `ScriptOptions` (all CLI configuration).
