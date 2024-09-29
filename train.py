import argparse
import json
from sketchdetection.datasets import SketchyDataset
from types import SimpleNamespace


def get_config(config_path):
    with open(config_path, "r") as config_file:
        return json.loads(config_file.read(), object_hook=lambda d: SimpleNamespace(**d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path of the training configuration JSON file', required=True)
    args = parser.parse_args()
    config = get_config(args.config)
