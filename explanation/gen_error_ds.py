import time

import openai
from openai import OpenAI
import os
import base64
import json
import re
import shutil
from tqdm import tqdm

from dataset import DATASET_ROUTES

os.environ["OPENAI_API_BASE"] = "https://api.pumpkinaigc.online/v1"
os.environ["OPENAI_BASE_URL"] = "https://api.pumpkinaigc.online/v1"
os.environ["OPENAI_API_KEY"] = "sk-xxxxxx"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_response(client, base64_image, prompt, model_name, max_tokens):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def response_check(response, img_name):
    # Response format check
    response = response.replace(" ", "")
    pattern = r'^(\d+)(,(\d+))*$'  # Match "1,3,6" format
    if not re.match(pattern, response):
        print(f"image: {img_name}, incorrect format response: {response}")
        return None

    # Rearrange indexes from low to high
    idx_list = sorted(list(map(int, response.split(','))))

    # Remove "Not Evident" when other categories match
    if 14 in idx_list and len(idx_list) > 1:
        idx_list.remove(14)

    if idx_list[0] < 1 or idx_list[-1] > 14:
        print(f"image: {img_name}, incorrect format response: {response}")
        return None

    # Index list to "1,3,6" format
    result = ','.join(map(str, idx_list))

    return result


def gen_classification_trial(dataset_dir, start: int=0, end: int=None):
    client = OpenAI(base_url='https://api.openai.com/v1')
    images = sorted(os.listdir(dataset_dir))

    for i, image_name in tqdm(enumerate(images)):
        if i < start:
            continue

        try:
            image_path = os.path.join(dataset_dir, image_name)
            base64_image = encode_image(image_path)

            prompt = """
                        Is the following image AI-generated? 
                        Respond "0" for synthetic images and "1" for natural images. Only reply "0" or "1", do not explain why.
                    """

            response = get_response(client, base64_image, prompt, model_name="gpt-4o", max_tokens=50)

            with open("output_cls.json", "a") as file:
                file.write(json.dumps({"idx": i, "img": image_name, "category": response}) + "\n")

        except (openai.InternalServerError, openai.APIConnectionError, openai.BadRequestError, FileNotFoundError) as e:
            print(f"Error with {image_name}, skipping. Error: {e}")
            time.sleep(5)
            continue


def gen_error_dataset(dataset_dir, target_dir, start: int=0, end: int=None):
    client = OpenAI(base_url='https://api.openai.com/v1')
    images = sorted(os.listdir(dataset_dir))
    if end is not None:
        images = images[start:end]

    # Target dirs for each error category
    for idx in range(1, 15):
        os.makedirs(os.path.join(target_dir, str(idx)), exist_ok=True)

    results = []
    for i, image_name in tqdm(enumerate(images)):
        try:
            image_path = os.path.join(dataset_dir, image_name)
            base64_image = encode_image(image_path)

            prompt = """
                        You will be provided with an AI generated image, try to identify systematic errors that make it
                        distinguishable from natural real images from the categories below:

                        1. Lighting: Unnatural or inconsistent light sources and shadows.
                        2. Color Saturation or Contrast: Overly bright or dull colors, extreme contrasts disrupting image harmony.
                        3. Perspective: Spatial disorientation caused by unrealistic angles or viewpoints, otherwise dimensionality errors such as flattened 3D objects.
                        4. Bad Anatomy: For living creatures, mismatches and errors of body parts in humans or animals.
                        5. Distorted Objects: For nonliving objects only, warped objects with fallacious details deviating from expected forms.
                        6. Structural Composition: Poor positional arrangement between multiple elements in the scene.
                        7. Incomprehensible Text: Malformed and unrecognizable text.
                        8. Implausible Scenarios: Inappropriate behavior and situations unlikely to happen based on sociocultural concerns, or contradicting to historical facts.
                        9. Physical Law Violations: Improbable physics, such as erroneous reflections or objects defying gravity.

                        Only select the categories below when highly confident:

                        10. Blurry or Inconsistent Borders: Unclear or abrupt border outlines between elements.
                        11. Background: Poorly blended or monotonous drab backgrounds.
                        12. Texture: For nonliving objects only, significantly over-polished or unnatural textural appearances.
                        13. Generation Failures: Major prominent rendering glitches or incomplete objects disrupting the entire scene.

                        If NONE of the categories above match, select:

                        14. Not Evident

                        Choose one or more categories above, and only reply the indexes of identified errors separated with
                        commas, e.g. "1,3,6" for an image with "Lighting, Perspective, Structural Composition" errors, with
                        no additional explanation.
                        PLEASE NOTE: Responses MUST be in the format of "[Number 1],[Number 2],..." (numbers and commas only)
                        ordered from low to high, or "14" if no evident errors are found.
                    """

            response = get_response(client, base64_image, prompt, model_name="gpt-4o", max_tokens=50)

            category = response_check(response, image_name)
            if category is None:
                print(f"Error with {image_name}, skipping")
                continue

            # Copy image to corresponding category directory
            for idx in category.split(","):
                target_path = os.path.join(target_dir, idx, image_name)
                shutil.copy(image_path, target_path)

        except (openai.InternalServerError, openai.APIConnectionError, FileNotFoundError) as e:
            print(f"Error with {image_name}, skipping")
            time.sleep(10)
            continue

        results.append({"img": image_name, "category": category})

    with open("./results/output.json", "w") as file:
        for item in results:
            file.write(json.dumps(item) + "\n")


def gen_explanation_dataset(err_category_dir):
    client = OpenAI(base_url='https://api.openai.com/v1')

    category_names = [
        "Lighting",
        "Color Saturation or Contrast",
        "Perspective",
        "Bad Anatomy",
        "Distorted Objects",
        "Structural Composition",
        "Incomprehensible Text",
        "Implausible Scenarios",
        "Physical Law Violations",
        "Blurry or Inconsistent Borders",
        "Background",
        "Texture",
        "Generation Failures",
    ]

    category_descriptions = [
        "Lighting: Unnatural or inconsistent light sources and shadows.",
        "Color Saturation or Contrast: Overly bright or dull colors, extreme contrasts disrupting image harmony.",
        "Perspective: Spatial disorientation caused by unrealistic angles or viewpoints, otherwise dimensionality errors such as flattened 3D objects.",
        "Bad Anatomy: For living creatures, mismatches and errors of body parts in humans or animals.",
        "Distorted Objects: For nonliving objects only, warped objects with fallacious details deviating from expected forms.",
        "Structural Composition: Poor positional arrangement between multiple elements in the scene.",
        "Incomprehensible Text: Malformed and unrecognizable text.",
        "Implausible Scenarios: Inappropriate behavior and situations unlikely to happen based on sociocultural concerns, or contradicting to historical facts.",
        "Physical Law Violations: Improbable physics, such as erroneous reflections or objects defying gravity.",
        "Blurry or Inconsistent Borders: Unclear or abrupt border outlines between elements.",
        "Background: Poorly blended or monotonous drab backgrounds.",
        "Texture: For nonliving objects only, significantly over-polished or unnatural textural appearances.",
        "Generation Failures: Major prominent rendering glitches or incomplete objects disrupting the entire scene.",
    ]

    for category_id in range(1, 14):
        category_dir = os.path.join(err_category_dir, str(category_id))
        images = sorted(os.listdir(category_dir))

        for image_name in tqdm(images):
            try:
                image_path = os.path.join(category_dir, image_name)
                base64_image = encode_image(image_path)

                prompt = f"""
                    You will be provided with an AI generated image confirmed with the following systematic error:
                    {category_descriptions[category_id - 1]}

                    Give an explanation for why such an error is found in the image, and point out the specific 
                    location or items causing the error. Pay close attention to image details. 
                    PLEASE NOTE: Responses should be concise, and organized into a SINGLE PARAGRAPH.
                """

                response = get_response(client, base64_image, prompt, model_name="gpt-4o-mini", max_tokens=300)
                item = {"category_id": category_id, "category_name": category_names[category_id - 1], "img": image_name, "explanation": response}
                with open(os.path.join(err_category_dir, "explanations.json"), "a") as file:
                    file.write(json.dumps(item) + "\n")

            except (openai.BadRequestError, openai.InternalServerError, openai.APIConnectionError, FileNotFoundError) as e:
                print(f"Error with {image_name}, skipping")
                time.sleep(10)
                continue



if __name__ == "__main__":
    ai_dataset_dir = f"{DATASET_ROUTES['SDV1.4']}/val/ai/"
    error_category_dir = f"{DATASET_ROUTES['SDV1.4']}/val/error-types/"
    gen_error_dataset(ai_dataset_dir, error_category_dir, 0, 6000)

    gen_explanation_dataset(f"{DATASET_ROUTES['SDV1.4']}/val/error-types-manual/")

    gen_classification_trial(f"{DATASET_ROUTES['GLIDE']}/val/ai/")
    gen_classification_trial(f"{DATASET_ROUTES['GLIDE']}/val/nature/")
