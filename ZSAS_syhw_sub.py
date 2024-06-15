# setting
import sys
import os
from datetime import datetime
sys.path.append('./SegmentAnything/GroundingDINO')
sys.path.append('./SegmentAnything/SAM')
sys.path.append('./SegmentAnything')
sys.path.append('./llama3')

import random
import csv
import argparse
from typing import List
import cv2
import numpy as np
import requests
import stringprep
import torch
import torchvision
import torchvision.transforms as TS
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionInpaintPipeline
from io import BytesIO
from matplotlib import pyplot as plt
from torchvision.ops import box_convert
import torchvision.ops as ops
from llama import Llama, Dialog
from ram import inference_ram
from ram.models import ram
import supervision as sv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from segment_anything import SamPredictor, build_sam, build_sam_hq
import SegmentAnything.SAA as SegmentAnyAnomaly
import GSA.GroundingDINO.groundingdino.datasets.transforms as T
from GSA.GroundingDINO.groundingdino.models import build_model
from GSA.GroundingDINO.groundingdino.util import box_ops
from GSA.GroundingDINO.groundingdino.util.inference import annotate
from GSA.GroundingDINO.groundingdino.util.slconfig import SLConfig
from GSA.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from ZSAS_funtion import show_mask, draw_mask, draw_box, load_image, load_model, get_grounding_output, \
    anomaly_llama, dilate_bounding_box, dilate_segment_mask, GroundedSAM, inpainting, find_largest_box_size, eval_zsas


# ArgumentParser 
parser = argparse.ArgumentParser(description='Description of your program')

parser.add_argument('--gpu', type=str, default="0", help='gpu_number')
parser.add_argument('--dataset', type=str, default="mvtec", help='dataset_name')
parser.add_argument('--main', type=str, default="hazelnut", help='main_name')
parser.add_argument('--sub', type=str, default="hazelnut", help='sub_name')
parser.add_argument('--db', type=float, default=1.2, help='Dilation bounding box')
parser.add_argument('--ds', type=int, default=5, help='Dilaton segment mask')
parser.add_argument('--ipa_diff_threshold', type=int, default=15, help='inpainting_diff_threshold')
parser.add_argument('--box_threshold', type=float, default=0.2, help='GroundingSAM box_threshold')
parser.add_argument('--text_threshold', type=float, default=0.2, help='GroundingSAM text_threshold')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='iou_threshold')
parser.add_argument('--temperature', type=float, default=0.6, help='Description of temperature argument')
parser.add_argument('--top_p', type=float, default=0.9, help='Description of top_p argument')
parser.add_argument('--max_seq_len', type=int, default=8192, help='Description of max_seq_len argument')
parser.add_argument('--max_gen_len', type=int, default=64, help='Description of max_gen_len argument')
parser.add_argument('--max_batch_size', type=int, default=4, help='Description of max_batch_size argument')

args = parser.parse_args()

gpu_number = args.gpu
db = args.db
ds = args.ds
inpainting_diff_threshold = args.ipa_diff_threshold
box_threshold = args.box_threshold
text_threshold = args.text_threshold
iou_threshold = args.iou_threshold
temperature = args.temperature
top_p = args.top_p
max_seq_len = args.max_seq_len
max_gen_len = args.max_gen_len
max_batch_size = args.max_batch_size
dataset_name = args.dataset
main_name = args.main
sub_name = args.sub

cr_date = datetime.now().strftime("%Y-%m-%d")

# load model
DEVICE = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else 'cpu')
SELECT_SAM_HQ = False

dino_config_file = "./GSA/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" 
dino_checkpoint = "./checkpoints/groundingdino_swint_ogc.pth"  
sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
sam_hq_checkpoint = "./checkpoints/sam_hq_vit_h.pth"
ram_checkpoint = "./checkpoints/ram_swin_large_14m.pth"
llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llama_api_token = "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"

saa_model = SegmentAnyAnomaly.Model(
    dino_config_file=dino_config_file,
    dino_checkpoint=dino_checkpoint,
    sam_checkpoint=sam_checkpoint,
    box_threshold=0.2,
    text_threshold=1.0,
    out_size=1024,
    device=DEVICE,
    ).to(DEVICE)

grounding_dino_model = load_model(dino_config_file, dino_checkpoint, DEVICE)

if SELECT_SAM_HQ:
    sam_model = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(DEVICE))
else:
    sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(DEVICE))

ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
ram_model.eval()
ram_model = ram_model.to(DEVICE)

login(llama_api_token)
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("------------MODEL LOAD COMPLETE------------")

folder_path = './dataset/{}/{}/test/{}'.format(dataset_name, main_name, sub_name)
gt_folder_path = './dataset/{}/{}/ground_truth/{}'.format(dataset_name, main_name, sub_name)
sub_numbers = os.listdir(folder_path)

saa_values = []
naive_values = []
ram_llama_values = []
syhw_values = []
    
for sub_number in sub_numbers:
    print('--------------------{} test is starting...-------------------------'.format(sub_number))
    image_path = os.path.join(folder_path, sub_number)
    gt_path = os.path.join(gt_folder_path, '{}_mask.png'.format(sub_number.split('.')[0]))
    print(image_path)
    print(gt_path)

    source_image, raw_image, ram_image, image, gt_image = load_image(image_path, gt_path)
    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    _, gt_binary = cv2.threshold(gt_image, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
    gt_mask = torch.tensor(gt_binary, dtype=torch.float)

    size_tuple = find_largest_box_size(grounding_dino_model, image, raw_image, main_name, box_threshold, text_threshold, iou_threshold, DEVICE)
    size_threshold = [size_tuple[0] - 0.01, size_tuple[1] - 0.01]
    print(size_threshold)

    # naive prompt
    NAIVE_TEXT_PROMPT = "defect"
    masks, boxes_filt, pred_phrases, _ = GroundedSAM(grounding_dino_model, sam_model, 
                                                    image, source_image, raw_image, NAIVE_TEXT_PROMPT, DEVICE,
                                                    box_threshold, text_threshold, iou_threshold)
    if len(masks) > 0:
        box_image = raw_image.copy()
        box_draw = ImageDraw.Draw(box_image)
        for box, label in zip(boxes_filt, pred_phrases):
            draw_box(box, box_draw, label)
        box_image_show = np.array(box_image)

        mask_image = Image.new('RGBA', (raw_image.size[0], raw_image.size[1]), color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
        mask_image_show = np.array(mask_image)

        naive_mask = sum(masks[i][0] for i in range(len(masks)))
        naive_mask = naive_mask > 0

        plt.figure(figsize=(15, 7))
        plt.subplot(141)
        plt.imshow(raw_image)
        plt.axis('off') 
        plt.title('Anomaly Image')
        plt.subplot(142)
        plt.imshow(box_image_show)
        plt.axis('off') 
        plt.title('Anomaly Detection')
        plt.subplot(143)
        plt.imshow(raw_image)
        plt.imshow(mask_image_show)
        plt.axis('off') 
        plt.title('Anomaly Segmentation')
        plt.subplot(144)
        plt.imshow(gt_image)
        plt.axis('off') 
        plt.title('GT')
        plt.savefig('./results_image_sy/naive/{}_naive_{}_{}_{}_{}'.format(cr_date, dataset_name, main_name, sub_name, sub_number), bbox_inches='tight', pad_inches=0.1)
    else:
        print("No masks found for the naive model")
        naive_mask = raw_image


    # ram + llama
    res = inference_ram(ram_image.to(DEVICE), ram_model)
    tags = res[0].strip(' ').replace('  ', ' ').replace(' |', ',')
    print('RAM finished')

    while True:
        llama_tags = anomaly_llama(llama_tokenizer, llama_model, tags) 
        print('Llama3 finished')

        RAM_LLAMA_TEXT_PROMPT = llama_tags
        masks, boxes_filt, pred_phrases, _ = GroundedSAM(grounding_dino_model, sam_model, 
                                                      image, source_image, raw_image, RAM_LLAMA_TEXT_PROMPT, DEVICE,
                                                      box_threshold, text_threshold, iou_threshold, size_threshold) 
        if masks is not None:  # GroundedSAM 함수가 성공적으로 값을 반환하면 루프 종료
            break
        print("Llama Error occurred in GroundedSAM. Retrying...") 
    
    box_image = raw_image.copy()
    box_draw = ImageDraw.Draw(box_image)
    for box, label in zip(boxes_filt, pred_phrases):
        draw_box(box, box_draw, label)
    box_image_show = np.array(box_image)

    mask_image = Image.new('RGBA', (raw_image.size[0], raw_image.size[1]), color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
    mask_image_show = np.array(mask_image)

    ram_llm_mask = sum(masks[i][0] for i in range(len(masks)))
    ram_llm_mask = ram_llm_mask > 0

    plt.figure(figsize=(15, 7))
    plt.subplot(141)
    plt.imshow(raw_image)
    plt.axis('off') 
    plt.title('Anomaly Image')
    plt.subplot(142)
    plt.imshow(box_image_show)
    plt.axis('off') 
    plt.title('Anomaly Detection')
    plt.subplot(143)
    plt.imshow(raw_image)
    plt.imshow(mask_image_show)
    plt.axis('off') 
    plt.title('Anomaly Segmentation')
    plt.subplot(144)
    plt.imshow(gt_image)
    plt.axis('off') 
    plt.title('GT')
    plt.savefig('./results_image_sy/ram+llm/{}_ram_llm_{}_{}_{}_{}'.format(cr_date, dataset_name, main_name, sub_name, sub_number), bbox_inches='tight', pad_inches=0.1)

    # ram + llama + ds +inpainting
    while True:
        masks, boxes_filt, pred_phrases, scores_filt = GroundedSAM(grounding_dino_model, sam_model, 
                                                      image, source_image, raw_image, RAM_LLAMA_TEXT_PROMPT, DEVICE,
                                                      box_threshold, text_threshold, iou_threshold, size_threshold, filt_ds=ds) 
        if masks is not None:  # GroundedSAM 함수가 성공적으로 값을 반환하면 루프 종료
            break
        print("Llama Error occurred in GroundedSAM. Retrying...") 
        
    inpainting_image, anomaly_map = inpainting(source_image, image_path, DEVICE,
                                            boxes_filt, scores_filt, pred_phrases, masks, 
                                            main_name, sub_name, sub_number, 
                                            inpainting_diff_threshold)

    box_image = raw_image.copy()
    box_draw = ImageDraw.Draw(box_image)
    for box, label in zip(boxes_filt, pred_phrases):
        draw_box(box, box_draw, label)
    box_image_show = np.array(box_image)

    mask_image = Image.new('RGBA', (raw_image.size[0], raw_image.size[1]), color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
    mask_image_show = np.array(mask_image)

    inpainting_mask_image = Image.new('RGBA', (raw_image.size[0], raw_image.size[1]), color=(0, 0, 0, 0))
    inpainting_mask_draw = ImageDraw.Draw(inpainting_mask_image)
    draw_mask(anomaly_map, inpainting_mask_draw, random_color=True)
    inpainting_mask_image_show = np.array(inpainting_mask_image)

    ram_llm_ds_ipa_mask = anomaly_map
    ram_llm_ds_ipa_mask = ram_llm_ds_ipa_mask > 0

    plt.figure(figsize=(15, 7))
    plt.subplot(241)
    plt.imshow(raw_image)
    plt.axis('off') 
    plt.title('Anomaly Image')
    plt.subplot(242)
    plt.imshow(box_image_show)
    plt.axis('off') 
    plt.title('Anomaly Detection')
    plt.subplot(243)
    plt.imshow(raw_image)
    plt.imshow(mask_image_show)
    plt.axis('off') 
    plt.title('Anomaly Segmentation')
    plt.subplot(245)
    plt.imshow(inpainting_image)
    plt.axis('off') 
    plt.title('Inpainting')
    plt.subplot(246)
    plt.imshow(raw_image)
    plt.imshow(inpainting_mask_image_show)
    plt.axis('off') 
    plt.title('Inpainting Anomaly Segmentation')
    plt.subplot(247)
    plt.imshow(gt_image)
    plt.axis('off') 
    plt.title('GT')
    plt.savefig('./results_image_sy/ram+llm+ds+ipa/{}_ram_llm_ds_ipa_{}_{}_{}_{}'.format(cr_date, dataset_name, main_name, sub_name, sub_number), bbox_inches='tight', pad_inches=0.1)


    # saa + mvtec/visa prompts
    import mvtec_parameters as mvtec_param
    import visa_parameters as visa_param

    manul_prompts = {
        'visa': visa_param.manual_prompts,
        'mvtec': mvtec_param.manual_prompts,
    }

    property_prompts = {
        'visa': visa_param.property_prompts,
        'mvtec': mvtec_param.property_prompts,
    }

    manul = manul_prompts[dataset_name]
    property = property_prompts[dataset_name]

    # saa+
    textual_prompts = manul[main_name]
    property_text_prompts =  property[main_name]

    saa_image = cv2.imread(image_path)
    saa_model.set_ensemble_text_prompts(textual_prompts, verbose=False)
    saa_model.set_property_text_prompts(property_text_prompts, verbose=False)
    score, appendix = saa_model(saa_image)

    similarity_map = appendix['similarity_map']
    similarity_map = cv2.resize(similarity_map, (raw_image.size[0], raw_image.size[1]))

    saa_mask = cv2.resize(score, (raw_image.size[0], raw_image.size[1]))
    saa_mask = saa_mask > 0

    plt.figure(figsize=(15, 7))
    plt.subplot(141)
    plt.imshow(raw_image)
    plt.axis('off') 
    plt.title('Anomaly Image')
    plt.subplot(142)
    plt.imshow(raw_image)
    plt.imshow(similarity_map, alpha=0.4, cmap='jet')
    plt.axis('off') 
    plt.title('Saliency')
    plt.subplot(143)
    plt.imshow(raw_image)
    plt.imshow(score, alpha=0.4,cmap='jet')
    plt.axis('off') 
    plt.title('Anomaly Score')
    plt.subplot(144)
    plt.imshow(gt_image)
    plt.axis('off') 
    plt.title('GT')
    plt.savefig('./results_image_sy/saa+/{}_saa+_{}_{}_{}_{}'.format(cr_date, dataset_name, main_name, sub_name, sub_number), bbox_inches='tight', pad_inches=0.1)


    # compare image
    plt.figure(figsize=(12, 10))
    plt.subplot(231)
    plt.imshow(raw_image)
    plt.axis('off') 
    plt.title('Anomaly Image')
    plt.subplot(232)
    plt.imshow(gt_image)
    plt.axis('off') 
    plt.title('Ground Truth')
    plt.subplot(233)
    plt.imshow(saa_mask)
    plt.axis('off') 
    plt.title('SAA+ (SOTA)')
    plt.subplot(234)
    plt.imshow(naive_mask.cpu().numpy())
    plt.axis('off') 
    plt.title('naive prompt ("defect")')
    plt.subplot(235)
    plt.imshow(ram_llm_mask.cpu().numpy())
    plt.axis('off') 
    plt.title(f'RAM + LLM ')
    plt.subplot(236)
    plt.imshow(ram_llm_ds_ipa_mask)
    plt.axis('off') 
    plt.title(f'{ds} dilation seg + Inpainting')
    plt.savefig('./results_image_sy/compare/{}_compare_{}_{}_{}_{}'.format(cr_date, dataset_name, main_name, sub_name, sub_number), bbox_inches='tight', pad_inches=0.1)
    
    # evaluation
    saa_values.append(eval_zsas(gt_mask, saa_mask))
    naive_values.append(eval_zsas(gt_mask, naive_mask))
    ram_llama_values.append(eval_zsas(gt_mask, ram_llm_mask))
    syhw_values.append(eval_zsas(gt_mask, ram_llm_ds_ipa_mask))
        
print(('saa+', [round(sum(column) / len(column), 4) for column in zip(*saa_values)]))
print(('naive', [round(sum(column) / len(column), 4) for column in zip(*naive_values)]))
print(('ram+llama', [round(sum(column) / len(column), 4) for column in zip(*ram_llama_values)]))
print(('syhw', [round(sum(column) / len(column), 4) for column in zip(*syhw_values)]))