from __future__ import print_function
import os

from tqdm import tqdm    
import torch
import torch.utils.data as data
import numpy
from nltk.corpus import stopwords
import shutil
import json
import spacy
from explanation_refine import *

from PIL import Image
from transformers import CLIPProcessor, CLIPModel,AutoTokenizer
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional
os.environ["TOKENIZERS_PARALLELISM"] = "false"


stop_words = set(stopwords.words('english'))
stop_words.add('image')
nlp = spacy.load("en_core_web_sm")

device = "cuda"
all_dir = "your_dataset_dir"
start = 0


clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
od_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
od_model.eval()


def find_best_match(image_path):
    image = Image.open(image_path)

    with torch.no_grad():
        all_tokens = {}
        for i in tqdm(range(0,clip_tokenizer.vocab_size)):
            text = clip_tokenizer.decode(torch.tensor([i]))
            inputs = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = clip_model(**inputs)
            text_embeds = outputs["pool_text_embeds"]
            image_embeds = outputs["image_embeds"]
            similarity = torch.nn.functional.cosine_similarity(text_embeds, image_embeds,dim=0)
            all_tokens[text] = similarity
        
        k = 10
        topk = dict(sorted(all_tokens.items(), key=lambda item: item[1], reverse=True)[:k])
        print(topk)


def find_borders(image, image_name):
    with torch.no_grad():
        transform = torchvision.transforms.ToTensor() 
        
        image_tensor = transform(image).to(device)
        image_tensor = image_tensor.unsqueeze(0)
        predictions = od_model(image_tensor)
        
        boxes = predictions[0]['boxes'][:].to("cpu")
    
        cropped_images = [image.crop((int(x_min), int(y_min), int(x_max), int(y_max))) for x_min, y_min, x_max, y_max in boxes.numpy()]
        return cropped_images, boxes


def get_border_sim(image_embeddings, text_embedding):
    with torch.no_grad():
        temperature = 0.1
        all_sim = []
        
        for img in image_embeddings:
            similarity = torch.nn.functional.cosine_similarity(text_embedding, img, dim=0)
            all_sim.append(similarity)
        all_sim = torch.tensor(all_sim).to(device)
        all_sim = torch.nn.functional.normalize(all_sim, p=2, dim=0)
        all_sim = all_sim / temperature
        all_img_embed = torch.stack(image_embeddings)
        weights = torch.softmax(all_sim, dim=0)
        weighted_embed = torch.matmul(weights, all_img_embed)
        
        total_similarity = torch.nn.functional.cosine_similarity(text_embedding, weighted_embed, dim=0)
        best_index = torch.argmax(all_sim).item()
        
        return total_similarity,best_index


def extract_words(text):
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    meaningful_phrases = []
    for item in noun_phrases:
        words = item.split(" ")
        meaningful_words = [word for word in words if word.lower() not in stop_words]
        if len(meaningful_words) > 0:
            phrase = " ".join(meaningful_words)
            meaningful_phrases.append(phrase) 
   
    return meaningful_phrases


def get_image_embeddings(images):
    with torch.no_grad():
        embeddings = []
        inputs = clip_processor(text=['text'], images=images, return_tensors="pt", padding=True, truncation=True, input_data_format="channels_last").to(device)
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds.squeeze()
        embeddings = torch.unbind(image_embeds, dim=0)
        return embeddings
   

def get_word_embeddings(words,img):
    with torch.no_grad():
        embeddings = []
        inputs = clip_processor(text=words, images=[img], return_tensors="pt", padding=True, truncation=True,input_data_format="channels_last").to(device)
        outputs = clip_model(**inputs)
        text_embeddings = outputs.text_embeds.squeeze()
        embeddings = torch.unbind(text_embeddings, dim=0)
        return embeddings


def expl_eval(explanations,n):
    print(all_dir)
    output_list=[]
    
    for item in tqdm(explanations):
        if item['explanation'] is None:
            output_list.append({'all_score':None,'top5_average':None,'top10_average':None,'average':None})
        else:
            image_name = item['img'].rstrip('.png')
            text = item['explanation']
            image_type = item['category_id']
            
            image_path = f"{all_dir}/{image_type}/{image_name}.png"
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            cropped_images, boxes = find_borders(image, image_name)
            cropped_images.append(image)
            boxes = torch.vstack([boxes, torch.tensor([0,0, image.height, image.width])])
            
            words = extract_words(text)
            words = [word for word in words if '/' not in word]
            word_scores = {}
            
            image_embeddings = get_image_embeddings(cropped_images)
            word_embeddings = get_word_embeddings(words,image)
            
            for word, word_embedding in zip(words, word_embeddings):
                score, index = get_border_sim(image_embeddings, word_embedding)
                word_scores[word] = score.item()
               
            if len(word_scores)>0:
                top5_values = sorted(word_scores.values(), reverse=True)[:5]
                top5_average = sum(top5_values) / len(top5_values)
                top10_values = sorted(word_scores.values(), reverse=True)[:10]
                top10_average = sum(top10_values) / len(top10_values)
                average = sum(word_scores.values())/len(word_scores.values())
                
                output_list.append({'all_score':word_scores,'top5_average':top5_average,'top10_average':top10_average,'average':average})
    
    score_output_path = f"{all_dir}/explanation_scores_{n}.json"
    with open(score_output_path, 'w') as json_file:
        json.dump(output_list, json_file, indent=4)
    return output_list
    

if __name__ == '__main__':
    
    if start == 0:
        file_path = f"{all_dir}/explanations.json"
        with open(file_path, "r", encoding="utf-8") as f:
            origin_explanations = [json.loads(line) for line in f]
    else:
        file_path = f"{all_dir}/explanation_{start}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            origin_explanations = json.load(f)

        
    explanations = origin_explanations
    origin_scores = expl_eval(origin_explanations,start)
    
    scores = origin_scores
    for i in range(start + 1, 4):
        print(f'round{i}')
        new_explanations = expl_refine(scores, explanations, i)
        new_scores = expl_eval(new_explanations, i)
        
        output_path = f"{all_dir}/explanation_refinement_{i}.json"
    
        result = []
        new_new_explanations = []
        new_new_scores = []
        for expln, expl0, scoren, score0 in zip(new_explanations, explanations, new_scores, scores):
            if expln['explanation'] is not None:
                result.append({"category_id": expl0['category_id'], "category_name": category_names[int(expl0['category_id']) - 1], "img": expl0['img'], 'origin_explanation':expl0['explanation'],'origin_top10_average':score0['top10_average'],'origin_top5_average':score0['top5_average'],'origin_average':score0['average'],'refined_explanation':expln['explanation'],'refined_top10_average':scoren['top10_average'],'refined_top5_average':scoren['top5_average'],'refined_average':scoren['average']})
                new_new_explanations.append(expln)
                new_new_scores.append(scoren)
        
        with open(output_path, 'w') as json_file:
            json.dump(result, json_file, indent=4)
            
        explanations = new_new_explanations
        scores = new_new_scores