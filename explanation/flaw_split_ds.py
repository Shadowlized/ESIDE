import os
import json
import random
import torch
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import FlawClassifyDataset, GENEXPLAIN_DATASET_ROUTES

device = "cuda" if torch.cuda.is_available() else "cpu"


def traverse_directory(root_dir):
    """Traverse directory, return folder names and file names"""
    file_list = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_list.append({"folder": subdir, "filename": file})
    return file_list


def generate_json_lists(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"⚠️ Skipping: directory not found {dataset_path}")
        return None

    all_files = traverse_directory(dataset_path)
    ds_dict = {}

    for item in all_files:
        # Get directory name as class label
        cat = os.path.basename(item["folder"])
        filename = item["filename"]
        if filename in ds_dict:
            ds_dict[filename].append(cat)
        else:
            ds_dict[filename] = [cat]

    result_list = [{"filename": k, "label": v} for k, v in ds_dict.items()]
    random.shuffle(result_list)

    # Split Train:Val = 9:1
    split_index = int(0.9 * len(result_list))
    train_list = result_list[:split_index]
    val_list = result_list[split_index:]

    # Save {"filename": k, "label": v} train/val sets into json
    with open(os.path.join(dataset_path, "train_list.json"), "w") as f:
        json.dump(train_list, f)
    with open(os.path.join(dataset_path, "val_list.json"), "w") as f:
        json.dump(val_list, f)

    return dataset_path


def main():
    for dataset_path in tqdm(GENEXPLAIN_DATASET_ROUTES, desc="Processing Datasets"):
        # 1. Create dataset splits json lists
        data_path = generate_json_lists(dataset_path)

        if data_path:
            # 2. Transform into Dataset instances
            train_dataset = FlawClassifyDataset(
                device,
                os.path.join(data_path, "train_list.json"),
                data_path
            )
            val_dataset = FlawClassifyDataset(
                device,
                os.path.join(data_path, "val_list.json"),
                data_path
            )

            # 3. Serialize Datasets into .pt files
            torch.save(train_dataset, os.path.join(dataset_path, f"train.pt"))
            torch.save(val_dataset, os.path.join(dataset_path, f"val.pt"))


if __name__ == "__main__":
    main()

