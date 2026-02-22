import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


dataset_root = "/your_home_dir/GenImage"
genexplain_dataset_root = "/your_home_dir/GenExplain"

DATASET_ROUTES = {
    "GLIDE": f"{dataset_root}/glide/imagenet_glide",
    "VQDM": f"{dataset_root}/VQDM/imagenet_ai_0419_vqdm",
    "ADM": f"{dataset_root}/ADM/imagenet_ai_0508_adm",
    "BigGAN": f"{dataset_root}/BigGAN/imagenet_ai_0419_biggan",
    "Wukong": f"{dataset_root}/wukong/imagenet_ai_0424_wukong",
    "Midjourney": f"{dataset_root}/Midjourney/imagenet_midjourney",
    "SDV1.4": f"{dataset_root}/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4",
    "SDV1.5": f"{dataset_root}/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5"
}

GENEXPLAIN_DATASET_ROUTES = {
    "GLIDE": f"{genexplain_dataset_root}/glide/imagenet_glide",
    "VQDM": f"{genexplain_dataset_root}/VQDM/imagenet_ai_0419_vqdm",
    "ADM": f"{genexplain_dataset_root}/ADM/imagenet_ai_0508_adm",
    "BigGAN": f"{genexplain_dataset_root}/BigGAN/imagenet_ai_0419_biggan",
    "Wukong": f"{genexplain_dataset_root}/wukong/imagenet_ai_0424_wukong",
    "Midjourney": f"{genexplain_dataset_root}/Midjourney/imagenet_midjourney",
    "SDV1.4": f"{genexplain_dataset_root}/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4",
    "SDV1.5": f"{genexplain_dataset_root}/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5"
}


class ClipFeatureDataset(Dataset):
    def __init__(self, image_folder, device, compute_features=True):
        self.dataset = image_folder
        self.device = device
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.features = []  # Precompute CLIP features of all images of dataset and save
        self.labels = []

        # Freeze CLIP model
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        dataloader = DataLoader(image_folder, batch_size=40, num_workers=0, pin_memory=True)

        with torch.no_grad():
            for batch_imgs, batch_labels in tqdm(dataloader, desc="Precomputing CLIP features..."):
                if compute_features:
                    processed_images = self.clip_processor(images=batch_imgs, return_tensors="pt", padding=True,
                                                           do_rescale=False).pixel_values.to(device)
                    batch_features = self.clip_model.get_image_features(processed_images)
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                    self.features.append(batch_features.cpu())  # Move features back to cpu
                self.labels.extend(batch_labels)

        # Concat all features
        if compute_features:
            self.features = torch.cat(self.features, dim=0)
        else:
            self.features = torch.tensor([0] * len(self.labels))
        self.labels = torch.tensor(self.labels)

        del self.clip_model
        del self.clip_processor
        torch.cuda.empty_cache()


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
            "idx": idx
        }


# Resize
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class FlawClassifyDataset(Dataset):

    def __init__(self, device, dataset_list_path, dataset_root):
        super().__init__()
        self.device = device
        self.features = []
        self.labels = []
        with open(dataset_list_path, "r", encoding="utf-8") as json_file:
            self.dataset_list = json.load(json_file)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        clip_model.eval()

        batch_size = 40

        for i in tqdm(range(0, len(self.dataset_list), batch_size), desc="Processing batches"):
            batch_items = self.dataset_list[i : i + batch_size]
            batch_imgs = []
            batch_labels = []

            for item in batch_items:
                filename = item["filename"]
                cat = item["label"][0]
                with open(f"{dataset_root}/{cat}/{filename}", "rb") as f:
                    img = clip_transform(Image.open(f).convert("RGB"))
                    batch_imgs.append(img)
                    label = [0] * 14
                    for id in item["label"]:
                        label[int(id) - 1] = 1
                    batch_labels.append(label)

            batch_imgs = torch.stack(batch_imgs)

            processed_imgs = clip_processor(
                images=batch_imgs, return_tensors="pt", padding=True, do_rescale=False
            ).pixel_values.to(self.device)

            with torch.no_grad():
                batch_features = clip_model.get_image_features(processed_imgs)

            self.features.extend(batch_features.cpu().split(1, dim=0))  # Split into single features
            self.labels.extend(batch_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"features": self.features[idx], "label": self.labels[idx], "idx": idx}