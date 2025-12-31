import os

import PIL
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier import BinaryClassifier
from dataset import clip_transform, DATASET_ROUTES, ClipFeatureDataset


def gen_hard_images(dataset_dir: str, target_dir: str, model_path: str):
    model = BinaryClassifier(device, network='S').to(device)
    model.load_state_dict(load_file(model_path))
    model.eval()

    dataset = ClipFeatureDataset(
        ImageFolder(root=dataset_dir, transform=clip_transform),
        device=device
    )
    dataloader = DataLoader(dataset, batch_size=40, num_workers=4, shuffle=False, pin_memory=True)

    os.makedirs(target_dir, exist_ok=True)

    ic_cnt = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images..."):
            features = batch["features"].to(model.device, non_blocking=True)
            labels = batch["label"].to(model.device, non_blocking=True)
            indexes = batch["idx"]

            outputs = model(features)
            probabilities = outputs['logits'].squeeze()
            predicted_classes = (probabilities >= 0.5).int()

            # Find incorrect predictions
            incorrect_mask = predicted_classes.cpu() != labels.cpu()
            incorrect_indexes = indexes[incorrect_mask]
            incorrect_probs = probabilities[incorrect_mask]
            incorrect_preds = predicted_classes[incorrect_mask]
            incorrect_true_labels = labels[incorrect_mask]

            # Save incorrect predictions
            os.makedirs(os.path.join(target_dir, "ai"), exist_ok=True)
            os.makedirs(os.path.join(target_dir, "nature"), exist_ok=True)
            for idx, prob, pred, true_label in zip(incorrect_indexes, incorrect_probs, incorrect_preds, incorrect_true_labels):
                original_path = dataset.dataset.imgs[idx][0]
                if true_label.item() == 0:
                    class_label = "ai"
                else:
                    class_label = "nature"
                filename = os.path.basename(original_path)
                save_path = os.path.join(target_dir, class_label, filename)
                try:
                    Image.open(original_path).save(save_path)
                    ic_cnt += 1
                    print(f"Incorrect Predicted: {class_label}/{filename}, Prob {prob.item():.4f}")
                except PIL.UnidentifiedImageError:
                    continue

    print(f"Total {ic_cnt} incorrect predicted images, stored at {target_dir}")


def remove_corrupt_images(dataset_dir):
    for filename in tqdm(os.listdir(dataset_dir), desc="Removing corrupt files..."):
        try:
            img = Image.open(os.path.join(dataset_dir, filename))
            img.verify()
        except Exception as ex:
            print(f"Corrupt file: {filename}, Exception: {ex}")
            os.remove(os.path.join(dataset_dir, filename))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    remove_corrupt_images(f"{DATASET_ROUTES['SDV1.4']}/train/ai")
    remove_corrupt_images(f"{DATASET_ROUTES['SDV1.4']}/train/nature")
    model_path = f"./models/your_ckpt_dir/checkpoint-90/model.safetensors"
    gen_hard_images(
        dataset_dir=f"{DATASET_ROUTES['SDV1.4']}/train",
        target_dir=f"{DATASET_ROUTES['SDV1.4']}/train/hard-samples",
        model_path=model_path
    )
