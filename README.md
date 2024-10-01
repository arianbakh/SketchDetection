# Environment Setup

1. `conda env create -f conda.yaml`
2. `conda activate sketchdetection`
3. `poetry install`
4. `apt install p7zip-full`
5. If poetry stalls at any point, try `poetry config keyring.enabled false`

# Training

1. Download the [Sketchy dataset](https://drive.google.com/file/d/1z4--ToTXYb0-2cLuUWPYM5m7ST7Ob3Ck/view) and put it in the `data` directory
2. `7z x data/rendered_256x256.7z -o./data/`
3. (Optional) cleanup:
    - `rm data/rendered_256x256.7z`
    - `rm -rf data/256x256/photo`
4. `python train.py --config configs/sample-config.json --checkpoint checkpoints/resnet152-1e-3.pth`

# Usage

- `python gui.py --port 60001 --architecture ResNet152 --checkpoint checkpoints/resnet152-1e-3.pth`
