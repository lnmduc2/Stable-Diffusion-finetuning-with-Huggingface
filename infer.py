import pandas as pd
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import torch
import argparse
import random
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of an inference script.")
    parser.add_argument("--prompts_path", type=str, default=None, help="path to the file containing test prompts")
    parser.add_argument("--results_path", type=str, default=None, help="path of the folder to contain result images")
    parser.add_argument("--model_path", type=str, default=None, help="path of the folder containing pretrained model or Hugging Face ID of the model")
    return parser.parse_args()

def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    max_length = pipeline.tokenizer.model_max_length
    count_prompt = len(prompt.split(" "))
    count_negative_prompt = len(negative_prompt.split(" "))

    if count_prompt >= count_negative_prompt:
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=shape_max_length, return_tensors="pt").input_ids.to(device)
    else:
        negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length", max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

def main():
    args = parse_args()
    TEST_PROMPTS_PATH = args.prompts_path
    RESULTS_PATH = args.results_path
    
    # Load model
    model_id = args.model_path or "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.enable_sequential_cpu_offload()
    
    # Assuming the file is tab-separated
    df = pd.read_csv(TEST_PROMPTS_PATH, sep='@', header=0)

    
    # Now, you can access the columns
    for index, row in df.iterrows():
        file_name = row['file_name']
        prompt = row['prompt']
        negative_prompt = row['negative_prompt']

        
        prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, negative_prompt, "cuda")
        image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds).images[0]
        image.save(f"{RESULTS_PATH}/{file_name}")
        

if __name__ == "__main__":
    main()
