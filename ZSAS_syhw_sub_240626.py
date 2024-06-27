# setting
import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath('./SegmentAnything/GroundingDINO'))
sys.path.append(os.path.abspath('./SegmentAnything/SAM'))
sys.path.append(os.path.abspath('./SegmentAnything'))
sys.path.append(os.path.abspath('./llama3'))

import random
from typing import List
import cv2
import argparse
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
from ZSAS_funtion import draw_mask, draw_box, load_image, load_model, get_grounding_output, anomaly_llama,\
    dilate_bounding_box, eval_zsas, paste_cropped_image, adjectiveclause_llama

# ArgumentParser 
parser = argparse.ArgumentParser(description='Description of your program')

parser.add_argument('--gpu', type=str, default="0", help='gpu_number')
parser.add_argument('--dataset', type=str, default="mvtec", help='dataset_name')
parser.add_argument('--main', type=str, default="hazelnut", help='main_name')
parser.add_argument('--sub', type=str, default="hazelnut", help='sub_name')
parser.add_argument('--db', type=float, default=1.5, help='Dilation bounding box')
parser.add_argument('--st', type=float, default=0.5, help='Size threshold = max_area * st(%)')
# parser.add_argument('--ds', type=int, default=5, help='Dilaton segment mask')
# parser.add_argument('--ipa_diff_threshold', type=int, default=15, help='inpainting_diff_threshold')
parser.add_argument('--box_threshold', type=float, default=0.2, help='GroundingSAM box_threshold')
parser.add_argument('--text_threshold', type=float, default=0.2, help='GroundingSAM text_threshold')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='iou_threshold')
parser.add_argument('--temperature', type=float, default=0.6, help='Description of temperature argument')
parser.add_argument('--top_p', type=float, default=0.9, help='Description of top_p argument')
parser.add_argument('--max_seq_len', type=int, default=8192, help='Description of max_seq_len argument')
parser.add_argument('--max_gen_len', type=int, default=64, help='Description of max_gen_len argument')
parser.add_argument('--max_batch_size', type=int, default=4, help='Description of max_batch_size argument')
parser.add_argument('--iteration', type=int, default=3, help='Description of test iteration argument')

args = parser.parse_args()

gpu_number = args.gpu
db = args.db
st = args.st
# ds = args.ds
# inpainting_diff_threshold = args.ipa_diff_threshold
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
iteration = args.iteration

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
    text_threshold=0.2,
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
sub_numbers = sorted(os.listdir(folder_path))

saa_values_last = []
syhw_values_last = []

for itr in range(iteration):
    print('--------------------{} iteration is starting...-------------------------'.format(itr))
    
    saa_values = []
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
        
        # object 식별
        TEXT_PROMPT = main_name
        boxes_filt, pred_phrases, scores = get_grounding_output(grounding_dino_model, image, 
                                                                TEXT_PROMPT, box_threshold, text_threshold, DEVICE)
        size = raw_image.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        boxes_filt = boxes_filt.cpu()

        # nms
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        scores = [scores[idx] for idx in nms_idx]
        
        widths = (boxes_filt[:, 2] - boxes_filt[:, 0])
        heights = (boxes_filt[:, 3] - boxes_filt[:, 1])
        boxes_area = (widths * heights)
        
        # 가장 큰 box 선택
        max_area, max_index = torch.max(boxes_area, dim=0)
        boxes_filt = boxes_filt[max_index]
        pred_phrases = pred_phrases[max_index]
        scores = scores[max_index]
        size_threshold = max_area * st
        
        left, upper, right, lower = boxes_filt
        left = int(left)
        upper = int(upper)
        right = int(right)
        lower = int(lower)

        cropped_raw_image = raw_image.crop((left, upper, right, lower))
        cropped_gt_image = gt_image[upper:lower, left:right]
        
        cropped_source_image = np.asarray(cropped_raw_image)

        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ram_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(),
            normalize])
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

        cropped_ram_image = cropped_raw_image.resize((384, 384))
        cropped_ram_image = ram_transform(cropped_ram_image).unsqueeze(0)

        cropped_image, _ = transform(cropped_raw_image, None)  # 3, h, w
        
        tags = main_name
        llama_tags = ''
        for word in tags.split(', '):
            print(word)
            llama_tags = llama_tags + adjectiveclause_llama(llama_tokenizer, llama_model, word)    
        TEXT_PROMPT = llama_tags

        while True:
            boxes_filt, pred_phrases, scores = get_grounding_output(grounding_dino_model, cropped_image, 
                                                                    TEXT_PROMPT, box_threshold, text_threshold, DEVICE)
            if boxes_filt is not None:  
                break

        print('org boxes:', len(boxes_filt))
        
        size = cropped_raw_image.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()

        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        scores = scores[nms_idx]

        print('nms boxes:',  len(boxes_filt))

        if len(boxes_filt) != 1:
            widths = (boxes_filt[:, 2] - boxes_filt[:, 0])
            heights = (boxes_filt[:, 3] - boxes_filt[:, 1])
            filt_idx = torch.nonzero(widths*heights < size_threshold).squeeze(1)
            filt_size = torch.unique(filt_idx)

            if (len(filt_size) != 0) and (len(filt_size) != len(boxes_filt)):
                boxes_filt = boxes_filt[filt_size]
                pred_phrases = [pred_phrases[i] for i in filt_size]
                scores = scores[filt_size]

        print('size_filt boxes:', len(boxes_filt))
        
        if db != None:
            for i in range(boxes_filt.size(0)):
                x_min, y_min, x_max, y_max = boxes_filt[i].tolist()
                new_x_min, new_y_min, new_x_max, new_y_max = dilate_bounding_box(x_min, y_min, x_max, y_max, scale=db)
                boxes_filt[i] = torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max])        

            boxes_filt[:, [0, 2]] = boxes_filt[:, [0, 2]].clamp(0, W)
            boxes_filt[:, [1, 3]] = boxes_filt[:, [1, 3]].clamp(0, H)

        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, (H, W)).to(DEVICE)

        # run SAM
        sam_model.set_image(cropped_source_image)
        masks, _, _ = sam_model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(DEVICE),
            multimask_output=False,
        )

        if masks is None:
            # cropped 이미지 크기에 맞게 모두 False로 채운 numpy 배열 생성
            masks = np.zeros((H, W), dtype=bool)

        #Draw Box
        box_image = cropped_raw_image.copy()
        box_draw = ImageDraw.Draw(box_image)
        for box, label in zip(boxes_filt, pred_phrases):
            draw_box(box, box_draw, label)
        box_image_show = np.array(box_image)

        #Draw Mask
        mask_image = Image.new('RGBA', (cropped_raw_image.size[0], cropped_raw_image.size[1]), color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)
        mask_image_show = np.array(mask_image)
        #Sum Mask
        ram_llm_mask = sum(masks[i][0] for i in range(len(masks)))
        ram_llm_mask = (ram_llm_mask > 0).cpu().numpy()

        plt.figure(figsize=(15, 7))
        plt.subplot(151)
        plt.imshow(box_image_show)
        plt.axis('off') 
        plt.title('Anomaly Detection')
        plt.subplot(152)
        plt.imshow(cropped_raw_image)
        plt.imshow(mask_image_show)
        plt.axis('off') 
        plt.title('Anomaly Segmentation')
        plt.subplot(153)
        plt.imshow(ram_llm_mask)
        plt.axis('off') 
        plt.title('Segmentation Mask')
        plt.subplot(154)
        plt.imshow(raw_image)
        result_mask = np.array(paste_cropped_image(Image.fromarray(gt_image.astype(np.uint8)).copy(),
                                        Image.fromarray(ram_llm_mask.astype(np.uint8)), (left, upper)))
        plt.imshow(result_mask, alpha=0.5, cmap='jet')
        plt.axis('off') 
        plt.title('Restore Segmentation Mask')
        plt.subplot(155)
        plt.imshow(raw_image)
        plt.imshow(gt_image, alpha=0.5, cmap='jet')
        plt.axis('off') 
        plt.title('GT')
        plt.show()
        plt.savefig('./results_image_sy/syhw/{}_{}_syhw_{}_{}_{}_{}'.format(cr_date, itr, dataset_name, main_name, sub_name, sub_number), bbox_inches='tight', pad_inches=0.1)


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
        plt.imshow(raw_image)
        plt.imshow(gt_image, alpha=0.5, cmap='jet')
        plt.axis('off') 
        plt.title('GT')
        plt.savefig('./results_image_sy/saa+/{}_{}_saa+_{}_{}_{}_{}'.format(cr_date, itr, dataset_name, main_name, sub_name, sub_number), bbox_inches='tight', pad_inches=0.1)


        # compare image
        plt.figure(figsize=(12, 10))
        plt.subplot(141)
        plt.imshow(raw_image)
        plt.axis('off') 
        plt.title('Anomaly Image')
        plt.subplot(142)
        plt.imshow(gt_image)
        plt.axis('off') 
        plt.title('Ground Truth')
        plt.subplot(143)
        plt.imshow(saa_mask)
        plt.axis('off') 
        plt.title('SAA+ (SOTA)')
        plt.subplot(144)
        plt.imshow(result_mask)
        plt.axis('off') 
        plt.title('SYHW (Ours)')
        plt.savefig('./results_image_sy/compare/{}_{}_compare_{}_{}_{}_{}'.format(cr_date, itr, dataset_name, main_name, sub_name, sub_number), bbox_inches='tight', pad_inches=0.1)
        
        # evaluation
        saa_values.append(eval_zsas(gt_mask, (saa_mask > 0)))
        syhw_values.append(eval_zsas(gt_mask, result_mask))
    
    saa_values_last = saa_values_last + saa_values
    syhw_values_last = syhw_values_last + syhw_values
    
print(('saa+', [round(sum(column) / len(column), 4) for column in zip(*saa_values_last)]))
print(('syhw', [round(sum(column) / len(column), 4) for column in zip(*syhw_values_last)]))