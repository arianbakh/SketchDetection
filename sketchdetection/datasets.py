import cv2
import json
import numpy as np
import os
import random
from torch.utils.data import Dataset


DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SKETCHY_DIR = os.path.join(DATASETS_DIR, "256x256", "sketch")
METADATA_FILE_NAME = "metadata.json"
SKETCHY_METADATA_PATH = os.path.join(SKETCHY_DIR, METADATA_FILE_NAME)
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42
SKETCHY_CLASSES = 125
SKETCHY_INPUT_SHAPE = (256, 256)


def get_class_index_to_name_map():
    with open(SKETCHY_METADATA_PATH, "r") as metadata_file:
        metadata = json.loads(metadata_file.read())
    return metadata["class_index_to_name"]


class SketchyDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self._ensure_metadata()
        with open(SKETCHY_METADATA_PATH, "r") as metadata_file:
            metadata = json.loads(metadata_file.read())
        if split == "train":
            self.data = metadata["train_images"]
        else:
            self.data = []
            for image_info in metadata["val_images" if split == "val" else "test_images"]:
                for image_path in image_info["image_paths"]:
                    self.data.append({
                        "class_index": image_info["class_index"],
                        "image_paths": [image_path]
                    })
        random.seed(RANDOM_SEED)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.data[index]["class_index"]
        image_paths = self.data[index]["image_paths"]
        if self.split == "train":
            image_path = random.choice(image_paths)
        else:
            image_path = image_paths[0]
        image = cv2.imread(image_path)
        model_input = np.swapaxes(image, 0, -1)
        model_input = model_input.astype(np.float32)
        return model_input, label

    def _ensure_metadata(self):
        if not os.path.exists(SKETCHY_METADATA_PATH):
            random.seed(RANDOM_SEED)
            class_index_to_name = {}
            class_name_to_index = {}
            intermediate_dict = {}
            for augmentation_name in sorted(os.listdir(SKETCHY_DIR)):
                augmentation_dir = os.path.join(SKETCHY_DIR, augmentation_name)
                if os.path.isdir(augmentation_dir):
                    for class_index, class_name in enumerate(sorted(os.listdir(augmentation_dir))):
                        class_dir = os.path.join(augmentation_dir, class_name)
                        if os.path.isdir(class_dir):
                            class_index_to_name[class_index] = class_name
                            class_name_to_index[class_name] = class_index
                            if not class_name in intermediate_dict:
                                intermediate_dict[class_name] = {}
                            for image_name in sorted(os.listdir(class_dir)):
                                if image_name.endswith(".png"):
                                    if not image_name in intermediate_dict[class_name]:
                                        intermediate_dict[class_name][image_name] = {
                                            "class_index": class_index,
                                            "image_paths": []
                                        }
                                    image_path = os.path.join(class_dir, image_name)
                                    intermediate_dict[class_name][image_name]["image_paths"].append(image_path)
            train_images = []
            val_images = []
            test_images = []
            for class_name, class_data in intermediate_dict.items():
                class_images = [v for k, v in class_data.items()]
                random.shuffle(class_images)
                train_index = int((1 - VAL_RATIO - TEST_RATIO) * len(class_images))
                val_index = int((1 - TEST_RATIO) * len(class_images))
                class_train_images = class_images[:train_index]
                class_val_images =  class_images[train_index:val_index]
                class_test_images =  class_images[val_index:]
                train_images += class_train_images
                val_images += class_val_images
                test_images += class_test_images
            metadata = {
                'class_index_to_name': class_index_to_name,
                'class_name_to_index': class_name_to_index,
                'train_images': train_images,
                'val_images': val_images,
                'test_images': test_images
            }
            with open(SKETCHY_METADATA_PATH, "w") as metadata_file:
                metadata_file.write(json.dumps(metadata))
