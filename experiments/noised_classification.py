import os
import time
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torchvision.datasets import ImageFolder
from safetensors.torch import load_file
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import BinaryClassifier
from dataset import ClipFeatureDataset, clip_transform, DATASET_ROUTES


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits >= 0.5).astype(int)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def train(model, dataset_name, train_set_dir, test_set_dir, step_select: int):
    start = time.time()
    print("Start Time: " + str(time.localtime()))

    training_args = TrainingArguments(
        output_dir=f"./models/step{step_select}_{dataset_name}-{time.strftime('%Y%m%d_%H%M', time.localtime())}",
        seed=42,
        learning_rate=5e-4,
        weight_decay=0.01,
        per_device_train_batch_size=120,
        per_device_eval_batch_size=120,
        num_train_epochs=20,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        dataloader_pin_memory=False,
    )

    train_dataset = ClipFeatureDataset(
        ImageFolder(train_set_dir, transform=clip_transform),
        device=device
    )
    val_dataset = ClipFeatureDataset(
        ImageFolder(test_set_dir, transform=clip_transform),
        device=device
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # trainer.save_model(f"./models/step{step_select}")

    print("End Time: " + str(time.localtime()))
    print("Elapsed Secs: " + str(time.time() - start))

    start = time.time()
    print("Start Time: " + str(time.localtime()))

    metrics = trainer.evaluate()
    print(metrics)

    print("End Time: " + str(time.localtime()))
    print("Elapsed Secs: " + str(time.time() - start))


def test(model, test_set_dir, save_dir):
    model_path = f"{save_dir}/model.safetensors"
    model.load_state_dict(load_file(model_path))
    model.eval()

    eval_args = TrainingArguments(
        output_dir="./",
        evaluation_strategy="epoch",
        per_device_eval_batch_size=120,
        dataloader_pin_memory=False,
    )

    val_dataset = ClipFeatureDataset(
        ImageFolder(test_set_dir, transform=clip_transform),
        device=device
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print(metrics)


def img_classification(image_path: str, step_select: int):
    model = BinaryClassifier(device).to(device)

    model_path = f"./models/step{step_select}/model.safetensors"
    # model_path = f"./models/step0-20241206_1234/checkpoint-90/model.safetensors"
    model.load_state_dict(load_file(model_path))
    model.eval()

    image = clip_transform(Image.open(image_path).convert("RGB"))
    processed_image = clip_processor(images=image, return_tensors="pt", padding=True, do_rescale=False).pixel_values
    processed_image = processed_image.to(device)

    with torch.no_grad():
        features = clip_model.get_image_features(processed_image).squeeze(0)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(features)
        probabilities = output['logits']
        predicted_class = (probabilities >= 0.5).int()

    print(f"Predicted class: {predicted_class.item()}, prob {probabilities.item()}")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BinaryClassifier(device).to(device)

    mode = "train"
    if mode == "train":
        step_selected = 0
        model = BinaryClassifier(device, network='L').to(device)
        train(
            model,
            "glide_L",
            f"{DATASET_ROUTES['GLIDE']}/val/stepwise-noised/{step_selected}/train-small",
            f"{DATASET_ROUTES['GLIDE']}/val/stepwise-noised/{step_selected}/val-small",
            step_selected,
        )
    elif mode == "test":
        step_selected = 0
        model = BinaryClassifier(device, network='L').to(device)
        test(
            model,
            f"{DATASET_ROUTES['GLIDE']}/val/stepwise-noised/{step_selected}/val-small",
            f"./models/your_ckpt_output_dir/checkpoint-1800"
        )

    elif mode == "single":
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        for i in range(1, 10):
            step_selected = 24
            img_classification(f"/your_home_dir/GenImage/glide/imagenet_glide/val/stepwise-noised/{step_selected}/val-small/ai/output_540{i}_step_{step_selected}.png", step_selected)
            img_classification(f"/your_home_dir/GenImage/glide/imagenet_glide/val/stepwise-noised/{step_selected}/val-small/nature/output_540{i}_step_{step_selected}.png", step_selected)

