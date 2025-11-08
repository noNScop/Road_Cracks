import datetime
import random
import time
from pathlib import Path

import dataset
import matplotlib.pyplot as plt
import torch
import typesum as ts
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

import models


def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    train_times, train_vals = train_losses
    test_times, test_vals = test_losses
    plt.plot(train_times, train_vals, label="train")
    plt.plot(test_times, test_vals, label="test")
    plt.scatter(train_times, train_vals)
    plt.scatter(test_times, test_vals)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()


def train_loop(device, model, train_dataset, test_dataset):
    model = model.to(device)

    BATCH_SIZE = 16

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # lossfn - weighted cross entropy (high class imbalance)
    # weights computed by segmentation/class_stats.py
    class_weights = [0.034954172600585066, 0.9650458273994149]
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

    print(f"Training model: {model.name()}")
    # generate ID from datetime
    model_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # time, loss
    train_losses = ([], [])
    test_losses = ([], [])

    iteration = 0
    for epoch in range(50):
        # TRAIN
        model.train()
        tq = tqdm(train_loader, desc=f"training epoch {epoch + 1}...")
        for images_, masks_ in tq:
            images = images_.to(device)
            masks = masks_.to(device)
            optimizer.zero_grad()
            outputs = softmax(model(images), dim=1)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            tq.set_postfix(loss=loss.item())
            train_losses[0].append(iteration)
            train_losses[1].append(loss.item())
            iteration += 1

        # TEST
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images_, masks_ in tqdm(
                test_loader,
                desc=f"testing epoch {epoch + 1}...",
            ):
                images = images_.to(device)
                masks = masks_.to(device)
                outputs = softmax(model(images), dim=1)
                loss = loss_fn(outputs, masks)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses[0].append(iteration)
        test_losses[1].append(test_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {loss.item()}, Test Loss: {test_loss}")

        # save checkpoint
        torch.save(
            model.state_dict(),
            f"models/{model.name()}_{model_id}_epoch{epoch + 1}.pth",
        )

        plot_losses(train_losses, test_losses)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.CrackSegmentationModel()

    # take a subset of dataset for quick testing
    rng = random.Random(42)
    # mod 3 > 1 to avoid batch size 1 error
    train_examples = rng.sample(dataset.list_examples(Path("train")), 200)
    test_examples = rng.sample(dataset.list_examples(Path("test")), 40)

    print(f"{len(train_examples)=}, {len(test_examples)=}")

    train_dataset = dataset.CrackSegmentationDataset(train_examples)
    test_dataset = dataset.CrackSegmentationDataset(test_examples)

    train_loop(device, model, train_dataset, test_dataset)


main()
