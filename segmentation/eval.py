import sys
from pathlib import Path

import dataset
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

import models


def inference(model, images):
    # returns a binary mask
    with torch.no_grad():
        outputs = model(images)
        probs = softmax(outputs, dim=1)
        return torch.argmax(probs, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.CrackSegmentationModel().to(device)

# load a trained model checkpoint
model.load_state_dict(
    torch.load(
        sys.argv[1],
        map_location=device,
    ),
)
model.eval()

# take a subset of dataset for quick testing
test_examples = dataset.list_examples(Path("test"))[:200]
test_dataset = dataset.CrackSegmentationDataset(test_examples)


def evaluate_on_dataset(test_dataset):
    # eval iou
    total_iou = 0.0
    tp = 0  # true positives
    fp = 0  # false positives
    tn = 0  # true negatives
    fn = 0  # false negatives
    for i in tqdm(range(len(test_dataset)), desc="evaluation..."):
        image, mask = test_dataset[i]
        image = image.unsqueeze(0).to(device)  # add batch dimension
        pred_mask = inference(model, image)  # (1,W,H)
        true_mask = torch.argmax(mask, dim=0).unsqueeze(0).to(device)  # (1,W,H)

        intersection = (pred_mask & true_mask).float().sum((1, 2))
        union = (pred_mask | true_mask).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        total_iou += iou.item()

        # compute tp, fp, tn, fn
        tp += ((pred_mask == 1) & (true_mask == 1)).sum().item()
        fp += ((pred_mask == 1) & (true_mask == 0)).sum().item()
        tn += ((pred_mask == 0) & (true_mask == 0)).sum().item()
        fn += ((pred_mask == 0) & (true_mask == 1)).sum().item()

    avg_iou = total_iou / len(test_dataset)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    fscore = 2 * (precision * recall) / (precision + recall + 1e-6)
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-score: {fscore:.4f}")


evaluate_on_dataset(test_dataset)

# show some examples
for i in range(20):
    image, mask = test_dataset[i]
    image = image.unsqueeze(0).to(device)  # add batch dimension
    pred_mask = inference(model, image)
    print(f"Example {i}: predicted mask shape: {pred_mask.shape}")


    # show image, predicted mask, true mask (side by side)
    plt.figure(figsize=(12, 4))
    plt.suptitle(f"Example {test_dataset.paths[i]}")
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask.squeeze(0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("True Mask")
    plt.imshow(torch.argmax(mask, dim=0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()
