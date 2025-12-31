import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.util import ngrams
from collections import Counter
import numpy as np
import math
import json
from tqdm import tqdm
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM


def calculate_ppl(texts):
    device = torch.device("cuda")
    model_name = 'gpt2'
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    
    ppls = []
    batch_size = 1
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i: i + batch_size]
        encodings = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
        
        ppl = torch.exp(loss).cpu().item()
        ppls.append(ppl)
    return sum(ppls) / len(ppls)
    

def text_quality_analysis(corpus):
    words = [word.lower() for word in word_tokenize(corpus) if word.isalnum()]
    
    stats = {}
    total_words = len(words)
    unique_words = len(set(words))
    
    stats['ttr'] = unique_words / total_words if total_words > 0 else 0  
  
    word_counts = Counter(words)
    entropy_value = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())
    stats['entropy'] = entropy_value  / math.log2(total_words)

    return stats


corpus = """
The lighting error in the image is primarily evident in the way shadows and highlights are portrayed on the bread and butter slices. The bread appears to be illuminated from multiple, inconsistent light sources rather than a single coherent direction, causing the top crust to have an unnatural brightness that conflicts with the shaded areas underneath. Additionally, the butter slices have a stark contrast in lighting; the upper layer of butter is overly bright while the lower slice casts a shadow that doesn't seem to align with the apparent light source. This dissonance creates an artificial and flat appearance rather than the natural depth and texture one would expect in such an arrangement.
"""

file_path = "your_dataset_dir"+"explanation_refinement_3.json"
with open(file_path, "r", encoding="utf-8") as f:
    inputs = json.load(f)

result = []

all_origin_explanations = ""
all_refined_explanations = ""

for item in tqdm(inputs):
    all_origin_explanations += "\n" + item['origin_explanation']
    quality0 = {}
    quality0['average'] = item['origin_average']
    quality0['top5_average'] = item['origin_top5_average']
    quality0['top10_average'] = item['origin_top10_average']
    
    all_refined_explanations += "\n" + item['refined_explanation']
    quality1 = {}
    quality1['average'] = item['refined_average']
    quality1['top5_average'] = item['refined_top5_average']
    quality1['top10_average'] = item['refined_top10_average']
    
    result.append({'origin_quality':quality0,'refined_quality':quality1})

origin_quality = text_quality_analysis(all_origin_explanations)
refined_quality = text_quality_analysis(all_refined_explanations)

origin_ppl = calculate_ppl([item['origin_explanation'] for item in inputs])
refined_ppl = calculate_ppl([item['refined_explanation'] for item in inputs])

print(f"""origin_quality: 
      ttr : {origin_quality['ttr']}
      entropy: {origin_quality['entropy']}
      origin_average : {sum([item['origin_quality']['average'] for item in result])/len(result)}
      origin_top5_average : {sum([item['origin_quality']['top5_average'] for item in result])/len(result)}
      origin_top10_average : {sum([item['origin_quality']['top10_average'] for item in result])/len(result)}
      ppl : {origin_ppl}
      
      refined_quality: 
      ttr : {refined_quality['ttr']}
      entropy: {refined_quality['entropy']}
      refined_average : {sum([item['refined_quality']['average'] for item in result])/len(result)}
      refined_top5_average : {sum([item['refined_quality']['top5_average'] for item in result])/len(result)}
      refined_top10_average : {sum([item['refined_quality']['top10_average'] for item in result])/len(result)}
      ppl : {refined_ppl}
      """)