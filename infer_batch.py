import sys
# sys.path.insert(0, '/mmlabworkspace/Students/visedit/ZALOAI2023/banner_advertisement')
import pandas as pd
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a inference script.")
    parser.add_argument(
        "--batch", type=int, default=8, help="batch size"
    )
    parser.add_argument(
        "--prompts_path", type=str, default=None, help="path to the file include test prompts"
    )
    parser.add_argument(
        "--results_path", type=str, default=None, help="path of the folder will contain result_images"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="path of the folder will contain pretrained model or huggingface id of model"
    )

    args = parser.parse_args()
    return args


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def postprocess(texts):
    '''
        Input: texts
        Output: processed texts
    '''
    res = []
    for text in texts:
        text = text.replace("-", " ").replace(",", "").replace(".", "")
        res.append(text)
    return res

def group_samples(samples, batch_size):
    '''
        Input:
            samples: list of samples
            batch_size: batch size
        Output:
            list of batches. Each batch is a list
    '''
    batches = [samples[i: i + batch_size] for i in range(0, len(samples), batch_size)]
    return batches
def main():
    args = parse_args()
    BATCH_SIZE = args.batch
    TEST_PROMPTS_PATH = args.prompts_path
    RESULTS_PATH = args.results_path

    with open(TEST_PROMPTS_PATH, 'r') as file:
        lines = file.readlines()
        data = [line.strip().split(',', 1) for line in lines[1:]] 
        raw_data = pd.DataFrame(data, columns=['file_name', 'caption'])

    model_path = args.model_path
    pipe = StableDiffusionPipeline.from_pretrained(model_path)
    pipe.to("cuda")
    seed = 28657
    set_seed(seed)

    data_batches = group_samples(raw_data, BATCH_SIZE)

    for row in tqdm(data_batches):
        batch_prompts = list(row["caption"])
        batch_names = list(row["file_name"])

        generator = torch.Generator(device="cuda").manual_seed(seed)
        images = pipe(prompt=batch_prompts, width=1024, height=536, generator=generator, num_inference_steps=89).images

        for image, file_name in zip(images, batch_names):
            print(file_name)
            resized_image = image.resize((1024, 533), Image.LANCZOS)
            resized_image.save(f"{RESULTS_PATH}/{file_name}")

if __name__ == "__main__":
    main()