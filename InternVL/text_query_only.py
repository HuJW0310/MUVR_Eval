import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

import os
import json
import re
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
# path = 'OpenGVLab/InternVL2_5-8B'
path = 'OpenGVLab/InternVL2-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=False,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
generation_config = dict(max_new_tokens=1024, do_sample=False)

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    if max_frame == 1:
        frame_indices = [0]
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# video_path = '/VLLM_Eval/LLaVA-NeXT/playground/demo/xU25MMA2N4aVtYay.mp4'
# pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
# pixel_values = pixel_values.to(torch.bfloat16).cuda()
# video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
# question = video_prefix + 'Just remember this video and return OK.'
# # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# video_path = '/VLLM_Eval/LLaVA-NeXT/playground/demo/xU25MMA2N4aVtYay.mp4'
# pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
# pixel_values = pixel_values.to(torch.bfloat16).cuda()
# video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
# question = video_prefix + 'If this video is same as the last one, return [Y] else [N].'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

class VideoQAProcessor:
    def __init__(self, model, tokenizer, gen_config):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.history = None
        
    def reset_history(self):
        self.history = None
        
    @staticmethod
    def build_video_prefix(num_frames, base=0):
        return "".join([f"Frame{i+1+base}: <image>\n" for i in range(num_frames)])
    
    @staticmethod
    def parse_response(response):
        return 1 if 'Yes' in response else 0
    
    def process_video(self, query_path, target_path, item, is_first=True):
        # Load video frames
        pixel_values_t, num_patches_list_t = load_video(target_path, num_segments=6, max_num=1)
        pixel_values_t = pixel_values_t.to(torch.bfloat16).cuda()
        
        # Build question
        # question_template = ''
        video_prefix_t = self.build_video_prefix(len(num_patches_list_t), base=0)
        # Let's think step by step.
        
        # question = 'Query: ' + item['prompt_en'] + "\n" + video_prefix_t + "\nThis is a video. If any part of the video fits the Query above (similar person, news, scene or event), return Yes. If not, return No."
        
        if item['Tag'] != "":
            question = f"I will give you a text query and a video: [Query] and [Target]. Please determine whether any part of [Target] is slightly relevant to any part of [Query]. I will also provide [Tag] that [Target] (if relevant) must feature it.\n[Query]:\n" + item['prompt_en'] + "\n[Target]:\n" + video_prefix_t + "\n[Tag]:\n" + item['Tag'] +"\n[Output]:\nIf slightly relevant, return Yes. If not, return No."
        else:
            question = f"I will give you a text query and a video: [Query] and [Target]. Please determine whether any part of [Target] is slightly relevant to any part of [Query].\n[Query]:\n" + item['prompt_en'] + "\n[Target]:\n" + video_prefix_t + "\n[Output]:\nIf slightly relevant, return Yes. If not, return No."

        # if item['Tag'] != "":
        #     question = f"I will give you a text query and a video: [Query] and [Target]. Please determine whether any part of [Target] is slightly relevant to any part of [Query]. I will also provide [Tag] that [Target] (if relevant) must feature it.\n[Query]:\n" + item['prompt_en'] + "\n[Target]:\n" + video_prefix_t + "\n[Tag]:\n" + item['Tag'] +"\n[Output]:\nLet's think step by step. Print the analysis and in the end, if slightly relevant, return Yes. If not, return No."
        # else:
        #     question = f"I will give you a text query and a video: [Query] and [Target]. Please determine whether any part of [Target] is slightly relevant to any part of [Query].\n[Query]:\n" + item['prompt_en'] + "\n[Target]:\n" + video_prefix_t + "\n[Output]:\nLet's think step by step. Print the analysis and in the end, if slightly relevant, return Yes. If not, return No."
        
        # Model inference
        response, self.history = self.model.chat(
            self.tokenizer,
            pixel_values_t,
            question,
            self.gen_config,
            num_patches_list=num_patches_list_t,
            history=None if is_first else self.history,
            return_history=True
        )
        return question, response

class JSONLDataset(Dataset):
    def __init__(self, jsonl_path, video_folder):
        self.data = []
        jsonl_path = Path(jsonl_path)
        video_folder = Path(video_folder)
        
        if 'all' in jsonl_path.name:
            parent_dir = jsonl_path.parent
            splits = ['news', 'instane', 'others', 'region', 'dance']
            
            for split in splits:
                # Generate split JSONL file path
                split_filename = jsonl_path.name.replace('all', split)
                split_jsonl = parent_dir / split_filename
                
                if not split_jsonl.exists():
                    continue
                
                # Read split JSONL file
                with open(split_jsonl, 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        if 'Tag' not in item.keys():
                            item['Tag'] = ""
                        if item['Tag'] == "":
                            # Add video paths
                            item['query_path'] = str(video_folder.parent / split / f"{item['Query']}.mp4")
                            item['target_path'] = str(video_folder.parent / split / f"{item['Target']}.mp4")
                            self.data.append(item)
        else:
            # Read single JSONL file
            with open(jsonl_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    # Add video paths
                    item['query_path'] = str(video_folder / f"{item['Query']}.mp4")
                    item['target_path'] = str(video_folder / f"{item['Target']}.mp4")
                    self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_results(results):
    # 处理空结果
    if not results:
        return {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "pair_metrics": {
                "ratios": {"+1": 0.0, "0": 0.0, "-1": 0.0, "-2": 0.0},
                "weighted_score": 0.0
            }
        }

    # 1. 计算准确率
    correct = sum(1 for x in results if x["pred"] == x["label"])
    accuracy = correct / len(results) * 100

    # 2. 计算F1分数
    tp = fp = fn = tn = 0
    for item in results:
        pred = item["pred"]
        label = item["label"]
        if label == 1:
            if pred == 1:
                tp += 1
            else:
                fn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 3. 计算配对指标
    pair_counts = {"+1": 0, "0": 0, "-1": 0, "-2": 0}
    total_pairs = len(results) // 2
    
    for i in range(0, len(results) - 1, 2):  # 保证成对处理
        pred1 = results[i]["pred"]
        pred2 = results[i+1]["pred"]
        
        if (pred1, pred2) == (1, 0):
            pair_counts["+1"] += 1
        elif (pred1, pred2) == (1, 1):
            pair_counts["0"] += 1
        elif (pred1, pred2) == (0, 0):
            pair_counts["-1"] += 1
        elif (pred1, pred2) == (0, 1):
            pair_counts["-2"] += 1

    # 计算比例和加权得分
    ratios = {}
    weighted_score = 0.0
    if total_pairs > 0:
        for key in pair_counts:
            ratios[key] = pair_counts[key] / total_pairs
        weighted_score = (
            pair_counts["+1"] * 1 +
            pair_counts["0"] * 0 +
            pair_counts["-1"] * (-1) +
            pair_counts["-2"] * (-2)
        ) / total_pairs

    return {
        "accuracy": accuracy,
        # "f1_score": f1,
        "pair_metrics": {
            "ratios": ratios,
            "weighted_score": weighted_score
        }
    }

def explain_tag(tag):
    if tag == "":
        return ""
    if tag[0] == "+":
        # return "[Target] must exist: " + tag[1:]
        return "must exist in [Target]: " + tag[1:]
    else:
        # return "[Target] must not exist: " + tag[1:]
        return "must not exist in [Target]: " + tag[1:]

def main():
    parser = argparse.ArgumentParser()
    split = 'all'
    parser.add_argument("--video_folder", type=str, default=f'/datasets/VR/M2VR/video_6_336/all_file/{split}')
    parser.add_argument("--jsonl_path", type=str, default=f'/s2vs/WebVR_Rerank/{split}_20.jsonl')
    # parser.add_argument("--jsonl_path", type=str, default=f'/s2vs/WebVR_Rerank/{split}_tv_0.05_clip.jsonl')
    
    
    # Q2 = "Describe this video and compare it with last video. If this video is similar to last video (same news or event), return Yes. If not, return No."
    # Q2 = "If B is similar to A (same news or event), return Yes. If not, return No."
    
    args = parser.parse_args()

    # Initialize model components
    # (假设model, tokenizer, generation_config已预先加载)
    processor = VideoQAProcessor(model, tokenizer, generation_config)
    dataset = JSONLDataset(args.jsonl_path, args.video_folder)

    results = []
    with tqdm(total=len(dataset), desc="Processing Videos", unit="step", 
             bar_format="{l_bar}{bar:20}{r_bar}", dynamic_ncols=True) as pbar:
        
        for item in dataset:
            # 第一阶段：视频比对
            if split == 'news':
                item['split_prompt'] = "similar event, scene or news"
            elif split == 'geng':
                item['split_prompt'] = "similar scene or pattern"
            elif split == 'animal':
                item['split_prompt'] = "similar instance like product, car or animal"
            elif split == 'region':
                item['split_prompt'] = "similar region or building"
            elif split == 'dance':
                item['split_prompt'] = "similar dance moving"
            
            item['Tag'] = explain_tag(item['Tag'])
            q1, r1 = processor.process_video(item['query_path'], item['target_path'], item, is_first=True)
            pred = processor.parse_response(r1)
            results.append({
                "query": item["Query"],
                "target": item["Target"],
                "tag": item["Tag"],
                "pred": pred,
                "label": item["Label"],
                "q1": q1,
                "r1": r1
            })
            pbar.update(1)
            pbar.set_postfix_str(f"Pred: {pred} | Label: {item['Label']}")
                
    # 计算准确率
    evaluate = evaluate_results(results)

    # 输出结果
    print("\n\nEvaluation Results:")
    print(f"Total Samples: {len(results)}")
    # print(f"Accuracy: {accuracy:.2f}%")
    print(evaluate)
    print("\nDetailed Predictions:")
    print("-"*60)
    for res in results[:2]:  # 展示前2个样例
        print(f"Pred: {res['pred']} | Label: {res['label']}")
        print(f"Q1: {res['q1'][:]}...\n")
        print(f"R1: {res['r1'][:]}...\n")
        print("-"*60)

if __name__ == "__main__":
    main()