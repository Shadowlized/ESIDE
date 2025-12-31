import json
import openai
from tqdm import tqdm
import os
import base64
from openai import OpenAI
import time

API_KEY = "sk-xxxxxx"

os.environ["OPENAI_API_KEY"] = API_KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = 'gpt-4o-mini'
client = OpenAI(base_url='https://api.openai.com/v1')

all_dir = "your_dataset_dir"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_response(client, base64_image, prompt, model_name, max_tokens):
    response = None
    try:
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
    except (openai.InternalServerError, openai.APIConnectionError, FileNotFoundError , openai.BadRequestError, openai.PermissionDeniedError,Exception) as e:
        print(f'errorr occurs: {e}')
        return None

            
    return response.choices[0].message.content

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

def expl_refine(scores,explanations,n):
    results = []
    for (score_item,explanation_item) in tqdm(zip(scores,explanations)):
        word_scores = score_item['all_score']
        highest_10 = dict(sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:10])
    
        image_name = explanation_item['img']
        explanation_text = explanation_item['explanation']
        category_id = explanation_item['category_id']
        
        image_path = f'{all_dir}/{category_id}/{image_name}'
        
        base64_image = encode_image(image_path)
    
       
        prompt = f"""You will be provided with an AI generated image confirmed with {category_names[category_id - 1]} error. Below is an explanation of why this error appears in the image:\n{explanation_text}\n In this explanation, the following words may be highly relevant and accurately describe the image: \n{', '.join(highest_10.keys())}\n Your task is to refine the explanation to better align with the error in the image while retaining the relevant words. You can also analyze the image to identify whether any other potential errors that may have been overlooked about {category_names[category_id - 1]}. Keep the explanation concise and avoid redundancy.\nPLEASE NOTE: Responses should be concise, and organized into a SINGLE PARAGRAPH.
        """

        response = get_response(client, base64_image, prompt, model_name=OPENAI_MODEL, max_tokens=300)
        results.append({"category_id": category_id, "category_name": category_names[category_id - 1], "img": image_name, 'explanation':response})

    output_path = f"{all_dir}/explanation_{n}.json"
    
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    
    return results 
