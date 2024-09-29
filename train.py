import argparse
import json
from sketchdetection.datasets import SketchyDataset
from sketchdetection.models import get_model
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace


def get_config(config_path):
    with open(config_path, "r") as config_file:
        return json.loads(config_file.read(), object_hook=lambda d: SimpleNamespace(**d))


def train(config):
    train_dataset = SketchyDataset("train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    model = get_model()
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.learning_rate_decay)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(config.epochs):
        for model_inputs, labels in tqdm(train_loader):
            model_inputs = model_inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            predictions = model(model_inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path of the training configuration JSON file', required=True)
    args = parser.parse_args()
    config = get_config(args.config)
    train(config)
