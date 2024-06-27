import sys
import os
# sys.path.append('./SegmentAnything/GroundingDINO')
# sys.path.append('./SegmentAnything/SAM')
# sys.path.append('./SegmentAnything')
# sys.path.append('./llama3')

sys.path.append(os.path.abspath('./SegmentAnything/GroundingDINO'))
sys.path.append(os.path.abspath('./SegmentAnything/SAM'))
sys.path.append(os.path.abspath('./SegmentAnything'))
sys.path.append(os.path.abspath('./llama3'))

import random
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

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.cpu().numpy().reshape(h, w, 1) * color.reshape(1, 1, -1)  # 수정된 부분
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)
        
def draw_box(box, draw, label):
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    line_width = int(max(4, min(20, 0.006 * max(draw.im.size))))

    # Draw rectangle
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color,  width=line_width)

    if label:
        font_path = os.path.join(
            cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        font_size = int(max(12, min(60, 0.02*max(draw.im.size))))
        font = ImageFont.truetype(font_path, size=font_size)
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white", font=font)

def load_image(image_path, gt_path):
    # load image
    raw_image = Image.open(image_path).convert("RGB")  # load image
    source_image = np.asarray(raw_image)

    gt_image = Image.open(gt_path).convert("RGB") 

    normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ram_transform = TS.Compose([
        TS.Resize((384, 384)),
        TS.ToTensor(),
        normalize
    ])

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    ram_image = raw_image.resize((384, 384))
    ram_image = ram_transform(ram_image).unsqueeze(0)

    image, _ = transform(raw_image, None)  # 3, h, w

    return source_image, raw_image, ram_image, image, gt_image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device, with_logits=True):
    if isinstance(caption, list):
        caption = ' '.join(caption)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()

    # filt_mask = logits_filt.max(dim=1)[0] > box_threshold1
    filt_mask = (logits_filt.max(dim=1)[0] > box_threshold) 
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):

        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer)

        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())
    # print(pred_phrases)
    
    return boxes_filt, pred_phrases, torch.Tensor(scores)

def anomaly_llama(tokenizer, model, tags):    
    messages = [{"role": "system", "content": "The assistant should always answer only by listing lowercase words in the following format: 'word, word'."},
                {"role": "user", "content": f"""Below is a list of objects recognized in the image: {tags}. Using each recognized object tag, we attempt to detect unusual or unusual parts of that object.

                Based on each recognized object tag, please create a list by converting it into tags that identify abnormal or unusual parts of the object.

                Please use adjectives or negatives to convert them into tags that indicate something unusual or strange.

                Additionally, each tag can be converted to multiple results."""},
            ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]

    # print('Tags : ', tokenizer.decode(response, skip_special_tokens=True))

    return tokenizer.decode(response, skip_special_tokens=True) 

def dilate_bounding_box(x_min, y_min, x_max, y_max, scale=1.0):
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    width = x_max - x_min
    height = y_max - y_min
    
    new_width = width * scale
    new_height = height * scale
    
    new_x_min = cx - new_width / 2
    new_y_min = cy - new_height / 2
    new_x_max = cx + new_width / 2
    new_y_max = cy + new_height / 2
    
    return new_x_min, new_y_min, new_x_max, new_y_max

def dilate_segment_mask(mask, kernel_size=5, iterations=1):
    """
    SAM에서 출력된 segmentation mask를 넓히는 함수

    :param mask: 이진 세그멘테이션 마스크 (numpy array)
    :param kernel_size: 커널 크기, 기본값은 5
    :param iterations: 팽창 연산 반복 횟수, 기본값은 1
    :return: 넓어진 세그멘테이션 마스크 (numpy array)
    """
    
    # 팽창 연산 커널
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    return dilated_mask

def GroundedSAM(grounding_dino_model, sam_model, 
                image, source_image, raw_image, tags, device,
                box_threshold, text_threshold, iou_threshold, size_threshold=None, filt_db=None, filt_ds=None):
    
    while True:
        boxes_filt, pred_phrases, scores = get_grounding_output(grounding_dino_model, image, 
                                                                tags, box_threshold, text_threshold, device)
        if boxes_filt is not None:  # GroundedSAM 함수가 성공적으로 값을 반환하면 루프 종료
            break

    # run SAM
    sam_model.set_image(source_image)
    size = raw_image.size

    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    scores = [scores[idx] for idx in nms_idx]
    
    if size_threshold is not None and len(boxes_filt) > 1:
        box_widths = (boxes_filt[:, 2] - boxes_filt[:, 0])/W # x_max - x_min
        box_heights = (boxes_filt[:, 3] - boxes_filt[:, 1])/H  # y_max - y_min

        # size_threshold의 각 값을 사용하여 조건에 맞는 인덱스를 찾음
        filt1_idx = torch.nonzero(box_widths < size_threshold[0]).squeeze(1)
        filt2_idx = torch.nonzero(box_heights < size_threshold[1]).squeeze(1)
        combined_indices = torch.cat((filt1_idx, filt2_idx))
        filt_size = torch.unique(combined_indices)

        if len(filt_size) != len(boxes_filt):
            boxes_filt = boxes_filt[filt_size]
            pred_phrases = [pred_phrases[i] for i in filt_size]
            scores = [scores[i] for i in filt_size]

    if filt_db != None:
        for i in range(boxes_filt.size(0)):
            x_min, y_min, x_max, y_max = boxes_filt[i].tolist()
            new_x_min, new_y_min, new_x_max, new_y_max = dilate_bounding_box(x_min, y_min, x_max, y_max, scale=filt_db)
            boxes_filt[i] = torch.tensor([new_x_min, new_y_min, new_x_max, new_y_max])        
        
        boxes_filt[:, [0, 2]] = boxes_filt[:, [0, 2]].clamp(0, W)
        boxes_filt[:, [1, 3]] = boxes_filt[:, [1, 3]].clamp(0, H)
        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, (H, W)).to(device)
    else:
        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, (H, W)).to(device)

    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    
    if masks is None:
        masks = boxes_filt
    
    if filt_ds != None:
        for i in range(len(masks)):
            dil = dilate_segment_mask(masks[i][0].cpu().numpy().astype(np.uint8), kernel_size=filt_ds, iterations=1)
            masks[i][0] = torch.tensor(dil > 0)
    
    return masks, boxes_filt, pred_phrases, scores

def inpainting(image, image_path, device,
               boxes_filt, scores_filt, pred_phrases, masks, 
               main_name, sub_name, sub_number, 
               inpainting_diff_threshold, filt_db=None, filt_ds=None):

    # Set Pipe
    if device.type == 'cpu':
        float_type = torch.float32
    else:
        float_type = torch.float16

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=float_type,
        ).to(device)

    inpainting_mask = sum(masks[i][0] for i in range(len(masks)))
    inpainting_mask = inpainting_mask > 0

    annotated_frame = annotate(image_source=image, boxes=boxes_filt, logits=scores_filt, phrases=pred_phrases)
    annotated_frame = annotated_frame[..., ::-1]

    image_mask = inpainting_mask.cpu().numpy()
    image_source_pil = Image.fromarray(image)
    image_mask_pil = Image.fromarray(image_mask)

    # annotated_frame_pil = Image.fromarray(annotated_frame)
    # annotated_frame_with_mask_pil = Image.fromarray(show_mask(inpainting_mask, annotated_frame))

    image_source_for_inpaint = image_source_pil.resize((512, 512))
    image_mask_for_inpaint = image_mask_pil.resize((512, 512))

    inpainting_image = pipe(prompt='', image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]   # prompt=main_name 제외
    inpainting_image = inpainting_image.resize((image_source_pil.size[0], image_source_pil.size[1]))

    ipa_path = "./results_image_sy/inpainting/ipa_{}_{}_{}_{}_{}.png".format(main_name, sub_name, sub_number, filt_db, filt_ds)
    inpainting_image.save(ipa_path)

    diff_raw_image = cv2.imread(image_path)
    diff_inpainted_image = cv2.imread(ipa_path)

    diff_image = cv2.absdiff(diff_raw_image, diff_inpainted_image)
    diff_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

    anomaly_map_1 = np.where(diff_gray > inpainting_diff_threshold, 255, 0)
    anomaly_map_2 = np.where(image_mask, anomaly_map_1, 0)

    return inpainting_image, anomaly_map_2

def remove_large_boxes(boxes, image_width, image_height):
    half_width, half_height = image_width / 2, image_height / 2

    mask = (boxes[:, 2] <= half_width) & (boxes[:, 3] <= half_height)
    filtered_boxes = boxes[mask]
    
    return filtered_boxes

def find_largest_box_size(grounding_dino_model, image, raw_image, tags,
                        box_threshold, text_threshold, iou_threshold, device):

    boxes_filt, pred_phrases, scores = get_grounding_output(
        grounding_dino_model, image, tags, box_threshold, text_threshold, device)

    size = raw_image.size
    H, W = size[1], size[0]

    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    scores = [scores[idx] for idx in nms_idx]

    widths = boxes_filt[:, 2] - boxes_filt[:, 0]
    heights = boxes_filt[:, 3] - boxes_filt[:, 1]

    normalized_widths = widths / W
    normalized_heights = heights / H
    
    largest_width = torch.max(normalized_widths)
    largest_height = torch.max(normalized_heights)
    
    return largest_width.item(), largest_height.item()

def eval_zsas(gt, pred_mask):
    if isinstance(gt, np.ndarray):
        gt_mask_np = gt
    else:
        gt_mask_np = gt.cpu().squeeze(0).numpy()
    
    if isinstance(pred_mask, np.ndarray):
        pred_mask_np = pred_mask
    else:
        pred_mask_np = pred_mask.cpu().squeeze(0).numpy()
    
    # Intersection over Union (IoU)
    intersection = np.logical_and(gt_mask_np, pred_mask_np)
    union = np.logical_or(gt_mask_np, pred_mask_np)
    iou = np.round(np.sum(intersection) / np.sum(union), 2)

    # Accuracy
    accuracy = np.sum(gt_mask_np == pred_mask_np) / gt_mask_np.size

    # Precision
    precision = np.sum(intersection) / np.sum(pred_mask_np)

    # Recall
    recall = np.sum(intersection) / np.sum(gt_mask_np)

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return iou, accuracy, precision, recall, f1_score

def paste_cropped_image(back_image, cropped_image, position):
    back_image.paste(cropped_image, position)
    return back_image

def add_word_to_each_item(word_list, word_to_add):
    words = word_list.split(', ')
    
    new_words = [word + ' ' + word_to_add for word in words]
    
    result = ', '.join(new_words)
    
    return result

def adjectiveclause_llama(tokenizer, model, tags):
    messages = [{"role": "system", "content": """The assistant should always answer only by listing lowercase words in the following format: 'word, word'."""},
                {"role": "user", "content": f"""The following objects are recognized in the image: {tags}.
                                                We want to create adjective clauses to prepend to object tags to find unusual or unusual aspects of the recognized object in the image.
                                                Here, ‘abnormal’ refers to a damaged or defective part of the product, or a part that is not visible under normal circumstances.
                                                Based on recognized object tags, adjectives or infinitives are converted to adjective clauses, creating a list that accurately specifies only the singular or unique parts of the object.
                                                Additionally, an adjective clause can necessarily be converted into 10 different results."""},
            ]

    with torch.no_grad():
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

    response = outputs[0][input_ids.shape[-1]:]

    result = tokenizer.decode(response, skip_special_tokens=True)

    finaly_result = add_word_to_each_item(result, tags)
    print(tags, ':', finaly_result)
    return finaly_result