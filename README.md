# wdv3-timm

small example thing showing how to use `timm` to run the WD Tagger V3 models.

## How To Use

1. clone the repository and enter the directory:
```sh
git clone https://github.com/deaquay/wdv3-timm.git
cd wd3-timm
```

2. Create a virtual environment and install the Python requirements.

If you're using Linux, you can use the provided script:
```sh
bash setup.sh
```

Or if you're on Windows (or just want to do it manually), you can do the following:
```sh
# Create virtual environment
python3.10 -m venv .venv
# Activate it
source .venv/bin/activate
# Upgrade pip/setuptools/wheel
python -m pip install -U pip setuptools wheel
# At this point, optionally you can install PyTorch manually (e.g. if you ARE using an nVidia GPU, cuXXX in url is cuda version, such as cu126 for cuda 12.6 check [Pytorch](https://pytorch.org/) for full install command.)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
# Install requirements
python -m pip install -r requirements.txt
```

3. Run the example script, with defaults (see -h/--help for info):
```sh
cd /path/wdv3_timm/
.venv/scripts/python wdv3_timm.py path/to/image.png
```
Or
```sh
cd /path/wdv3_timm/
.venv/scripts/python wdv3_timm.py path/to/folder
```

Example output from `.venv/scripts/python wdv3_timm.py a_picture_of_ganyu.png`:
```sh
Using tag file format: filename_tags.txt
Using cuda
Loading model 'eva02' from 'SmilingWolf/wd-eva02-large-tagger-v3'...
Loading tag list...
Creating data transform...
Processing single image...
--------
Caption: 1girl, horns, solo, bell, ahoge, colored_skin, blue_skin, neck_bell, looking_at_viewer, purple_eyes, upper_body, blonde_hair, long_hair, goat_horns, blue_hair, off_shoulder, sidelocks, bare_shoulders, alternate_costume, shirt, black_shirt, cowbell, ganyu_(genshin_impact)
--------
Tags: 1girl, horns, solo, bell, ahoge, colored skin, blue skin, neck bell, looking at viewer, purple eyes, upper body, blonde hair, long hair, goat horns, blue hair, off shoulder, sidelocks, bare shoulders, alternate costume, shirt, black shirt, cowbell, ganyu \(genshin impact\)
--------
Ratings:
  general: 0.827
  sensitive: 0.199
  questionable: 0.001
  explicit: 0.001
--------
Character tags (threshold=0.75):
  ganyu_(genshin_impact): 0.991
--------
General tags (threshold=0.35):
  1girl: 0.996
  horns: 0.950
  solo: 0.947
  bell: 0.918
  ahoge: 0.897
  colored_skin: 0.881
  blue_skin: 0.872
  neck_bell: 0.854
  looking_at_viewer: 0.817
  purple_eyes: 0.734
  upper_body: 0.615
  blonde_hair: 0.609
  long_hair: 0.607
  goat_horns: 0.524
  blue_hair: 0.496
  off_shoulder: 0.472
  sidelocks: 0.470
  bare_shoulders: 0.464
  alternate_costume: 0.437
  shirt: 0.427
  black_shirt: 0.417
  cowbell: 0.415
```
## --help contents
```
usage: wdv3_timm.py [-h] [-g GEN_THRESHOLD] [-c CHAR_THRESHOLD] [-b BATCH_SIZE] [-q] [-f] [-r] [-s SUFFIX] [-m {eva02,vit-large,vit,convnext,swinv2}] path

positional arguments:
  path                  Path to image file or directory

options:
  -h, --help            show this help message and exit
  -g GEN_THRESHOLD, --gen-threshold GEN_THRESHOLD
                        General tag threshold (default: 0.35)
  -c CHAR_THRESHOLD, --char-threshold CHAR_THRESHOLD
                        Character tag threshold (default: 0.75)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Number of images to process at once (default: 8)
  -q, --quiet           Run in quiet mode, output only tags, single image only.
  -f, --force           Process images even if they already have txt files
  -r, --recursive       Recursively search subdirectories for images
  -s SUFFIX, --suffix SUFFIX
                        Suffix for tag files: 'tags' for filename_tags.txt, 'none' for filename.txt (default: tags)
  -m {eva02,vit-large,vit,convnext,swinv2}, --model {eva02,vit-large,vit,convnext,swinv2}
                        Model to use (default: eva02)
```
