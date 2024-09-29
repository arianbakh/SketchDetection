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


def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for model_inputs, labels in tqdm(data_loader):
            model_inputs = model_inputs.cuda()
            labels = labels.cuda()
            predictions = model(model_inputs)
            correct += (torch.max(predictions, dim=1)[1] == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    model.train()
    return accuracy


def train(config, checkpoint_path):
    train_loader, val_loader, test_loader = [
        DataLoader(
            SketchyDataset(split),
            batch_size=config.batch_size,
            shuffle=True if split == "train" else False,
            num_workers=config.num_workers
        ) for split in ["train", "val", "test"]
    ]
    model = get_model()
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.learning_rate_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_accuracy = 0
    for epoch in range(config.epochs):
        running_correct = 0
        running_total = 0
        progress_bar = tqdm(train_loader)
        for model_inputs, labels in progress_bar:
            model_inputs = model_inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            predictions = model(model_inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            optimizer.step()
            running_correct += (torch.max(predictions, dim=1)[1] == labels).sum().item()
            running_total += labels.size(0)
        scheduler.step()
        train_accuracy = 100 * running_correct / running_total
        val_accuracy = calculate_accuracy(model, val_loader)
        saved = False
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            saved = True
        print(f"Epoch: {epoch + 1}/{config.epochs}; Train acc.: {train_accuracy:.2f}; Val acc.: {val_accuracy:.2f};{"saved" if saved else ""}")
    test_accuracy = calculate_accuracy(model, test_loader)
    print(f"Test acc. {test_accuracy:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path of the training configuration JSON file', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path of the checkpoint', required=True)
    args = parser.parse_args()
    config = get_config(args.config)
    train(config, args.checkpoint)
