import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import time
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import BinaryClassifier
from dataset import ClipFeatureDataset, clip_transform, DATASET_ROUTES


class WeightedBCELoss:
    def __init__(self):
        pass

    def __call__(self, logits, targets, sample_weights, n_samples):
        """
        Weighted Binary Cross Entropy Loss。

        Args:
            logits: Model outputs, Tensor, Shape: (n_samples,)
            targets: True labels, Tensor, Shape: (n_samples,)
            sample_weights: Weights assigned to each dataset sample, Tensor, Shape: (n_samples,)

        Returns:
            Weighted Loss, Original Loss
        """
        epsilon = 1e-7
        logits = torch.clamp(logits, epsilon, 1 - epsilon)  # logits -> (epsilon, 1-epsilon) to avoid log(0)

        # BCELoss
        losses = - (targets * torch.log(logits) + (1 - targets) * torch.log(1 - logits))  # Shape: (n_samples,)
        weighted_losses = losses * sample_weights  # Shape: (n_samples,)
        return n_samples * torch.mean(weighted_losses), torch.mean(losses)  # weighted_loss, original_loss



class AdaBoostWrapper:
    def __init__(self, model, optimizer, loss_fn, device, epochs=10):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs

    def fit(self, X, y, sample_weights, n_samples):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(X)["logits"].squeeze()
        loss, unweighted_loss = self.loss_fn(outputs, y, sample_weights, n_samples)
        loss.backward()
        self.optimizer.step()
        return loss.item(), unweighted_loss.item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)["logits"].squeeze()
            predictions = (outputs >= 0.5).int()
        return predictions



class CustomAdaBoost:
    def __init__(self, models, device, epochs=10, batch_size=120):
        self.models = models
        self.device = device
        self.alphas = [0 for _ in range(len(self.models))]    # Classifier weights assigned, higher accuracy higher alpha
        self.epochs = epochs
        self.batch_size = batch_size


    def fit(self, args):
        os.makedirs(args.save_dir, exist_ok=True)

        dataloader = None
        if args.run_degradation:
            # AdaBoost All Original Unnoised Images
            train_dataset = ClipFeatureDataset(
                ImageFolder(f"{args.dataset_root_dir}/0/{args.train_split}", transform=clip_transform),
                device=device
            )
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

        for i, model in enumerate(self.models):
            # Continue training, if loaded
            if i < args.continue_pretrain_model_idx:
                print(f"Skipping Model {i}...")
                continue

            if not args.run_degradation:
                # As each model uses a different dataset, assign N size of 1/N weights each iteration, for a total of len(models) * N size
                train_dataset = ClipFeatureDataset(
                    ImageFolder(f"{args.dataset_root_dir}/{i * args.step_size}/{args.train_split}", transform=clip_transform),
                    device=device
                )
                dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

            n_samples = len(dataloader) * self.batch_size  # Count training labels of images from a single step diffusion
            sample_weights = torch.ones(n_samples, dtype=torch.float32, device=self.device) / n_samples  # 1/N weight initially assigned to each training sample

            for _ in tqdm(range(self.epochs), desc=f"Training Model {i}"):
                epoch_alphas, epoch_weighted_losses, epoch_original_losses, epoch_errors = [], [], [], []

                for batch in dataloader:
                    X_train = batch["features"].to(model.device, non_blocking=True)
                    y_train = batch["label"].to(model.device, non_blocking=True)
                    indexes = batch["idx"].to(model.device, non_blocking=True)
                    current_weights = sample_weights[indexes]   # sample_weights[indexes] select weights corresponding to current batch

                    # Train model
                    weighted_loss, original_loss = model.fit(X_train, y_train, current_weights, n_samples)
                    epoch_weighted_losses.append(weighted_loss)
                    epoch_original_losses.append(original_loss)

                    # Calc error = sum of wrong predictions weights; y_pred & y_train & weights all of len(labels) size
                    y_pred = model.predict(X_train)
                    error = (torch.sum(current_weights * (y_pred != y_train).float()) / torch.sum(current_weights)).item()
                    epoch_errors.append(error)
                    if error < 0.001:
                        error = 0.001
                    if error > 0.5:
                        error = 0.499  # Keep min alpha = 0.002

                    # Calc alpha weight for current classifier, 0.5ln((1-err)/err)
                    alpha = 0.5 * torch.log(torch.tensor((1 - error) / error, device=self.device))
                    epoch_alphas.append(alpha.item())

                    # Projection: 0, 1 -> -1, 1
                    y_train = 2 * y_train - 1
                    y_pred = 2 * y_pred - 1
                    is_correct = y_train * y_pred

                    # Update new sample weights; Normalization, w_i = w_i / sum(w)
                    wlr = 0.25  # Set sample weight learning rate
                    current_weights *= torch.exp(- wlr * alpha * is_correct)  # Classified wrong then add weights
                    sample_weights[indexes] = current_weights

                self.alphas[i] = sum(epoch_alphas) / len(epoch_alphas)
                sample_weights /= torch.sum(sample_weights)  # Normalization

                avg_weighted_loss = sum(epoch_weighted_losses) / len(epoch_weighted_losses)
                avg_original_loss = sum(epoch_original_losses) / len(epoch_original_losses)
                avg_error = sum(epoch_errors) / len(epoch_errors)
                print(f"Epoch average weighted loss: {avg_weighted_loss:.4f}")
                print(f"Epoch average original loss: {avg_original_loss:.4f}")
                print(f"Epoch average error: {avg_error:.4f}")

        self.save_model(f"{args.save_dir}/{args.network_type}-{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.pth")  # Saves final model


    def ada_predict(self, args, dataset_dir):
        predictions = []

        dataloader = None
        if args.run_degradation:
            # AdaBoost All Original Unnoised Images
            val_dataset = ClipFeatureDataset(
                ImageFolder(f"{dataset_dir}/0/{args.val_split}", transform=clip_transform),
                device=device
            )
            dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        for i in range(len(self.models)):
            if not args.run_degradation:
                val_dataset = ClipFeatureDataset(
                    ImageFolder(f"{dataset_dir}/{i * args.step_size}/{args.val_split}", transform=clip_transform),
                    device=device
                )
                dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            preds = []
            for batch in tqdm(dataloader, desc="Test Batch"):
                X_test = batch["features"]
                X_test = X_test.to(self.models[i].device)
                with torch.no_grad():
                    outputs = self.models[i].model(X_test)["logits"].squeeze()
                    preds.append((outputs >= 0.5).int())

            predictions.append(self.alphas[i] * torch.cat(preds, dim=0))

        # Majority voting, total of sum(self.alphas) votes for each sample
        predictions = torch.stack(predictions, dim=0)  # (n_models, n_samples)
        final_predictions = (predictions.sum(dim=0) >= (0.5 * sum(self.alphas))).int()
        return final_predictions


    def save_model(self, filepath):
        save_dict = {
            "models": self.models,
            "alphas": self.alphas
        }
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")


    def load_model(self, filepath):
        ckpt = torch.load(filepath)
        self.models = ckpt["models"]
        self.alphas = ckpt["alphas"]
        print(f"Model loaded from {filepath}")


    def crop_model(self, step_size):
        """ Crop model to a larger step stride, when loading a stride=1 full model ensemble of Step0-Step24 models """
        self.models = [model for idx, model in enumerate(self.models) if idx % step_size == 0]
        self.alphas = [alpha for idx, alpha in enumerate(self.alphas) if idx % step_size == 0]


def main(args):
    start = time.time()
    print("Start Time: " + str(time.localtime()))

    models = []
    for i in tqdm(range(0, int(24 / args.step_size) + 1), desc="Initializing Models"):
        if args.run_degradation:
            # AdaBoost All Original Unnoised Images
            torch.manual_seed(args.seed + i)
            torch.cuda.manual_seed(args.seed + i)
            np.random.seed(args.seed + i)
        else:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)

        model = BinaryClassifier(device, network=args.network_type).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=5e-4)
        loss_fn = WeightedBCELoss()
        models.append(AdaBoostWrapper(model, optimizer, loss_fn, device, args.epochs))

    custom_adaboost = CustomAdaBoost(models, device, args.epochs, args.batch_size)

    if args.mode == "train":
        if args.continue_pretrain_model_idx != -1:
            custom_adaboost.load_model(f"{args.save_dir}/{args.ckpt_fn}")
        custom_adaboost.fit(args)
    elif args.mode == "test" and args.ckpt_fn != "":
        custom_adaboost.load_model(f"{args.save_dir}/{args.ckpt_fn}")
        # Only use Model 0:
        # custom_adaboost.alphas = [3.45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if len(custom_adaboost.models) == 25 and args.step_size != 1:
            custom_adaboost.crop_model(args.step_size)
    else:
        print("Mode incorrect or model checkpoint unassigned")
        quit(-1)

    eval_model(args, custom_adaboost, args.dataset_root_dir)
    if args.dataset_hard_root_dir is not None:
        eval_model(args, custom_adaboost, args.dataset_hard_root_dir)

    print("End Time: " + str(time.localtime()))
    print("Elapsed Secs: " + str(time.time() - start))


def eval_model(args, model, dataset_dir):
    y_pred = model.ada_predict(args, dataset_dir)
    y_pred = y_pred.to(device)

    val_dataset = ClipFeatureDataset(
        ImageFolder(f"{dataset_dir}/0/{args.val_split}", transform=clip_transform),
        device=device,
        compute_features=False
    )
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    y_test_gold = torch.cat([batch["label"] for batch in dataloader], dim=0).to(device)  # Gold labels of first dataset is enough

    correct_cnt = torch.sum(y_pred == y_test_gold).item()
    accuracy = correct_cnt / y_test_gold.size(0)
    print(f"Dataset: {dataset_dir}")
    print(f"Custom AdaBoost Test Accuracy: {accuracy:.5f}")
    print(f"Alphas: {model.alphas}")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    DATASETS = ["Midjourney", "SDV1.4", "SDV1.5", "ADM", "GLIDE", "VQDM", "Wukong", "BigGAN"]
    DATASET_NAME = "SDV1.4"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_dir", type=str, default=f"{DATASET_ROUTES[DATASET_NAME]}/val/stepwise-noised")
    parser.add_argument("--dataset_hard_root_dir", type=str, default=f"{DATASET_ROUTES[DATASET_NAME]}/train/hard-samples/stepwise-noised")
    parser.add_argument("--train_split", type=str, default="train-small")
    parser.add_argument("--val_split", type=str, default="val-small")
    parser.add_argument("--save_dir", type=str, default=f"./models/adaboost_{DATASET_NAME.replace('.', '')}_L_step3")
    parser.add_argument("--network_type", type=str, default="L")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--step_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--run_degradation", type=bool, default=False)
    parser.add_argument("--continue_pretrain_model_idx", type=int, default=-1)
    parser.add_argument("--ckpt_fn", type=str, default="")
    args = parser.parse_args()

    print(f"\n\n ------------- {args.network_type} Stride {args.step_size} SD V1.4 --------------\n\n")

    main(args)
