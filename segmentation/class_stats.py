from pathlib import Path

import dataset
from tqdm import tqdm

# Compute % of each class for loss weights


def compute_class_stats(dataset):
    class_counts = [0, 0]  # background, crack
    total_pixels = 0

    for i in tqdm(range(len(dataset))):
        _, mask = dataset[i]  # mask shape: (2,W,H)
        mask_np = mask.numpy()
        class_counts[0] += (mask_np[0] == 1).sum()
        class_counts[1] += (mask_np[1] == 1).sum()
        total_pixels += mask_np.shape[1] * mask_np.shape[2]

    class_percentages = [count / total_pixels for count in class_counts]
    return class_counts, class_percentages


if __name__ == "__main__":
    train_examples = dataset.list_examples(Path("train"))
    train_dataset = dataset.CrackSegmentationDataset(train_examples)

    class_counts, class_percentages = compute_class_stats(train_dataset)
    print(f"Class Counts: {class_counts}")
    print(f"Class Percentages: {class_percentages}")
