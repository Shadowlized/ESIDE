import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import math

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import DATASET_ROUTES


def uniform_from_two_ranges(a1, b1, a2, b2):
    if random.random() < 0.5:
        return random.uniform(a1, b1)
    else:
        return random.uniform(a2, b2)

def rotate_crop_and_resize(img, angle):
    w_orig, h_orig = img.size
    side = min(w_orig, h_orig)

    # Step 1: Crop center square
    left = (w_orig - side) // 2
    top = (h_orig - side) // 2
    img_square = img.crop((left, top, left + side, top + side))

    # Step 2: Rotation
    rotated = img_square.rotate(angle, expand=True)

    # Step 3: Safe center square
    radians = math.radians(abs(angle))
    sin_a = abs(math.sin(radians))
    cos_a = abs(math.cos(radians))
    side_safe = int(side / (sin_a + cos_a + 1e-6))  # 防止除零

    # Step 4: Crop to safe center square
    w_rot, h_rot = rotated.size
    left = (w_rot - side_safe) // 2
    top = (h_rot - side_safe) // 2
    cropped = rotated.crop((left, top, left + side_safe, top + side_safe))

    # Step 5: Resize to original
    final = cropped.resize((w_orig, h_orig), Image.BICUBIC)
    return final

def random_blur(img):
    radius = random.uniform(0.5, 2.5)
    return img.filter(ImageFilter.GaussianBlur(radius)), f"blur_radius_{radius:.2f}"

def random_rotate(img):
    angle = uniform_from_two_ranges(-45,-5,5,45)
    return rotate_crop_and_resize(img,angle), f"rotate_angle_{angle:.2f}"

def random_brightness(img):
    factor = uniform_from_two_ranges(0.3,0.9,1.1,1.8)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor), f"brightness_factor_{factor:.2f}"

def apply_random_perturbation(img):
    methods = [random_blur, random_rotate, random_brightness]
    method = random.choice(methods)
    return method(img)

def perturb_images_randomly(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            input_path = os.path.join(input_dir, filename)
            name, ext = os.path.splitext(filename)
            try:
                img = Image.open(input_path).convert("RGB")
                perturbed_img, desc = apply_random_perturbation(img)
                output_filename = f"{name}_{desc}{ext}"
                output_path = os.path.join(output_dir, output_filename)
                perturbed_img.save(output_path)
                print(f"{filename} -> {output_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


classes=["ai","nature"]
models=["ADM", "BigGAN", "GLIDE", "MidJourney", "SDV1.4", "SDV1.5", "VQDM", "Wukong"]

for model in models:
    for clss in classes:
        perturb_images_randomly(
            input_dir=f"{DATASET_ROUTES[model]}/val/stepwise-noised/0/val-small/{clss}",
            output_dir=f"{DATASET_ROUTES[model]}/val/stepwise-noised/0/pert/{clss}"
        )
