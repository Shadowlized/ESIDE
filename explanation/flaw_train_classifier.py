import os
import time
import torch
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, hamming_loss, average_precision_score
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import BinaryClassifier
from dataset import GENEXPLAIN_DATASET_ROUTES


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = labels.astype(int)

    naive_predictions = (logits >= 0.5).astype(int)

    naive_EM = accuracy_score(labels, naive_predictions)
    naive_hamming = hamming_loss(labels, naive_predictions)

    # calculate mAP
    ap_per_class = []
    for i in range(labels.shape[1]):
        ap = average_precision_score(labels[:, i], logits[:, i])
        ap_per_class.append(ap)
    map_score = sum(ap_per_class) / len(ap_per_class)

    return {"map": map_score, "naive_EM": naive_EM, "naive_hamming": naive_hamming}


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_list = ["L", "M"]

    for dataset_path in tqdm(GENEXPLAIN_DATASET_ROUTES):
        train_dataset = torch.load(os.path.join(dataset_path, f"train.pt"))
        val_dataset = torch.load(os.path.join(dataset_path, f"val.pt"))

        for network_size in model_list:
            print(f"start training...")
            print(f"dataset:{dataset_path}\nmodel:{network_size}")
            model = BinaryClassifier(device, network_size=network_size).to(device)
            training_args = TrainingArguments(
                output_dir=f"./models/{os.path.basename(dataset_path)}-{network_size}-{time.strftime('%Y%m%d_%H%M', time.localtime())}",
                seed=33,
                learning_rate=5e-4,
                weight_decay=0.01,
                per_device_train_batch_size=64,
                per_device_eval_batch_size=64,
                num_train_epochs=30,
                report_to="none",
                # logging_dir="./logs",
                # logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="naive_EM",
                dataloader_pin_memory=False,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            trainer.save_model(f"./models/{os.path.basename(dataset_path)}-{network_size}")
            metrics = trainer.evaluate()
            print(f"final result:")
            print(metrics)
