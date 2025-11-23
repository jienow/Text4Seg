import argparse
#from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
import torch
import json
from functools import partial
import cv2
import random
from itertools import combinations
# from datasets import load_from_disk
import tqdm
# import pdb
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from PIL import Image #  as PILImage
import re
# from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.question_answer_list import QUESTION_PARTIAL
from llava.conversation import conv_templates
from llava.mm_utils import process_images
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
import os

# LLaVA conversation模板
from llava.conversation import conv_templates, SeparatorStyle

# 常量：IMAGE_TOKEN、<im_start>、<im_end> 等
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)

# LLaVA 的图像处理 & tokenizer拼接方法
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path
)

# SAM predictor
from llava.model.segment_anything import (
    SamPredictor,
    sam_model_registry
)

# Text4Seg 工具函数：粗 mask → logits / 采点 / 解码序列
from llava.eval.utils import (
    compute_logits_from_mask,
    masks_sample_points,
    translate_sequence,
    decode_mask
)

# 参考表达生成模板
from llava.eval.question_answer_list import QUESTION_PARTIAL



ANSWER_LIST_PEST = [
    "是 [SEG]。",
    "分割结果是 [SEG]。",
    "[SEG]。",
]
ANSWER_LIST_MODE4_TEMPLATE_PEST = [
    "{class_name} [SEG]",
    "{class_name}:[SEG]",
    "{class_name}的掩码是[SEG]",
    "{class_name}的分割掩码是[SEG]",
    "{class_name}的指代是[SEG]"
]
ANSWER_LIST_MODE4_TEMPLATE_NON_PEST = [
    "{class_name}:[NON]",
]
ANSWER_LIST_MODE4_START_PEST = [
    "当然,",
    "当然,",
    "当然,"
]
ANSWER_LIST_MODE4_END = [
    "。", "。", "。", "。", "。"
]
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
class ValDatasetPest(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        val_dataset,
        image_size=1024,
        image_root_path=None,
        reason_seg_data_add=None,
        reason_image_root_path=None
    ):
        self.image_root_path = image_root_path
        self.reason_seg_data_add=reason_seg_data_add
        self.reason_image_root_path=reason_image_root_path
        self.base_image_dir = base_image_dir
        self.answer_list = ANSWER_LIST_PEST
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = []
            jsons = []
            if self.reason_seg_data_add is not None:
                whole_path = os.path.join(reason_seg_data_add, 'reason_seg', ds, "val")
                if not os.path.exists(whole_path):
                    print(f"路径不存在: {whole_path}")
                image_root_path = self.reason_image_root_path

                for root, dirs, files in os.walk(whole_path):
                    for file in files:
                        json_path = os.path.join(root, file)
                        assert os.path.exists(json_path)

                        jsons.append(json_path)
                        file_name = file.split(".")[0]
                        image_file_name = str(file_name) + ".jpg"

                        image_path = os.path.join(image_root_path, image_file_name)
                        if not os.path.exists(image_path):
                            image_root_path2 = "/home/luohuibin/pycharm_workspace/PixelLM-main/pest24_data/4k_image"
                            image_path = os.path.join(image_root_path2, image_file_name)
                            if not os.path.exists(image_path):
                                image_path = os.path.join(image_root_path2, str(file_name.replace("xt_", "")) + ".png")
                        assert os.path.exists(image_path)
                        images.append(image_path)
            else:
                whole_path = os.path.join(base_image_dir, 'reason_seg', ds, "val")
                if not os.path.exists(whole_path):
                    print(f"路径不存在: {whole_path}")

                image_root_path = self.image_root_path

                for root, dirs, files in os.walk(whole_path):
                    for file in files:
                        json_path = os.path.join(root, file)
                        assert os.path.exists(json_path)

                        jsons.append(json_path)
                        file_name = file.split(".")[0]
                        image_file_name = str(file_name) + ".jpg"

                        image_path = os.path.join(image_root_path, image_file_name)
                        if not os.path.exists(image_path):
                            image_root_path2 = "/home/luohuibin/pycharm_workspace/PixelLM-main/pest24_data/4k_image"
                            image_path = os.path.join(image_root_path2, image_file_name)
                            # if not os.path.exists(image_path):
                            #     image_path = os.path.join(image_root_path2, str(file_name.replace("xt_", "")) + ".png")
                        assert os.path.exists(image_path)
                        images.append(image_path)
            self.jsons = jsons
            self.images = images

            self.data_type = "reason_seg"

        self.ds = ds
        self.image_size = image_size
        # self.tokenizer = tokenizer
        # self.transform = ResizeLongestSide(image_size)
        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.class_name_answer = ANSWER_LIST_MODE4_TEMPLATE_PEST
        self.class_name_answer_non = ANSWER_LIST_MODE4_TEMPLATE_NON_PEST
    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)

            # output_path = "output_image_or1.jpg"
            # cv2.imwrite(output_path, image)

            image_cl = image
            image_cl = cv2.cvtColor(image_cl, cv2.COLOR_BGR2RGB)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # output_path = "output_image_last1.jpg"
            # cv2.imwrite(output_path, image)


            json_path = self.jsons[idx]
            # mask_json, sampled_sents, is_sentence, pest_list = get_mask_from_json_pest(json_path, image)

            mask_json, sampled_sents, is_sentence, pest_list,bbox_list,non_segment_list, non_pest_list, non_bbox_list = get_mask_from_json_pest_mul(json_path, image)

            if not is_sentence:
                all_bbox_list = bbox_list
                all_pest_list = pest_list
                all_mask_list = mask_json

                #用于决定挑选几只害虫作为目标
                one_choice = random.choice([1, 2, 3])
                if one_choice >= len(all_pest_list):
                    one_choice = 1
                print('one_choice:',one_choice)
                if one_choice == 2:
                   #选择两只害虫
                    index_list = [index for index, value in enumerate(pest_list)]
                    pest_combinations = list(combinations(index_list,2))
                    w_list = []
                    for pc in pest_combinations:
                        w_list.append(1)
                    assert len(w_list) == len(pest_combinations)
                    is_choice = random.choices(pest_combinations,weights=w_list,k=1)[0]

                    # 创建剩余的索引列表
                    non_mask_idx = [i for i in range(len(all_mask_list)) if i not in is_choice]
                    non_mask_list = []
                    for non_i in non_mask_idx:
                        non_mask_list.append(all_mask_list[non_i])
                        non_pest_list.append(all_pest_list[non_i])
                        non_bbox_list.append(all_bbox_list)
                    # 使用 reduce 函数按位逻辑“或”合并所有掩码
                    non_merged_mask_ = non_mask_list
                    # if len(non_mask_idx) != 0:
                    #     non_mask_list = []
                    #     for non_i in non_mask_idx:
                    #         non_mask_list.append(all_ms[non_i])
                    #     non_merged_mask_ = non_mask_list
                    # else:
                    #     non_merged_mask_ = []

                    mask_json = [mask_json[ic] for ic in is_choice]
                    pest_list = [pest_list[ic] for ic in is_choice]
                    bbox_list = [bbox_list[ic] for ic in is_choice]
                elif one_choice == 3:
                   #选择两只害虫
                    index_list = [index for index, value in enumerate(pest_list)]
                    pest_combinations = list(combinations(index_list,3))
                    w_list = []
                    for pc in pest_combinations:
                        w_list.append(1)
                    assert len(w_list) == len(pest_combinations)
                    is_choice = random.choices(pest_combinations,weights=w_list,k=1)[0]

                    # 创建剩余的索引列表
                    non_mask_idx = [i for i in range(len(all_mask_list)) if i not in is_choice]

                    non_mask_list = []
                    for non_i in non_mask_idx:
                        non_mask_list.append(all_mask_list[non_i])
                        non_pest_list.append(all_pest_list[non_i])
                        non_bbox_list.append(all_bbox_list)
                    # 使用 reduce 函数按位逻辑“或”合并所有掩码
                    non_merged_mask_ = non_mask_list

                    mask_json = [mask_json[ic] for ic in is_choice]
                    pest_list = [pest_list[ic] for ic in is_choice]
                    bbox_list = [bbox_list[ic] for ic in is_choice]

                elif one_choice == 1:
                    #选择一只害虫作为目标
                    is_choice = random.choice(range(len(pest_list)))
                    # 创建剩余的索引列表
                    non_mask_idx = [i for i in range(len(pest_list)) if i != is_choice]

                    non_mask_list = []
                    for non_i in non_mask_idx:
                        non_mask_list.append(all_mask_list[non_i])
                        non_pest_list.append(all_pest_list[non_i])
                        non_bbox_list.append(all_bbox_list)
                    # 使用 reduce 函数按位逻辑“或”合并所有掩码
                    non_merged_mask_ = non_mask_list
                    mask_json = [mask_json[is_choice]]
                    pest_list = [pest_list[is_choice]]
                    bbox_list = [bbox_list[is_choice]]

                sampled_sents = [sampled_sents]
            else:
                all_bbox_list = bbox_list + non_bbox_list
                all_pest_list = pest_list + non_pest_list
                all_mask_list = mask_json + non_segment_list


                assert len(non_segment_list) == len(non_bbox_list) == len(non_pest_list)
                if len(non_segment_list) != 0:
                    non_merged_mask_ = non_segment_list
                else:
                    non_merged_mask_ = []
                sampled_sents = sampled_sents



        conversations = []
        # conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            # conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                # conv.append_message(
                #     conv.roles[0],
                #     DEFAULT_IMAGE_TOKEN
                #     + "\n {} 请给出分割掩码。".format(text),
                # )

                answers = []
                ans_start = random.choice(ANSWER_LIST_MODE4_START_PEST)
                ans_end = random.choice(ANSWER_LIST_MODE4_END)
                seg_token_parts = []
                non_token_parts = []
                for pest_name in pest_list:
                    question_template = random.choice(self.class_name_answer)
                    question_template = question_template.format(class_name=pest_name)
                    seg_token_parts.append(question_template)

                for non_pest_name in non_pest_list:
                    ans_template = random.choice(self.class_name_answer_non)
                    ans_template = ans_template.format(class_name=non_pest_name)
                    non_token_parts.append(ans_template)
                #使用多个seg
                # answers.append(
                #     ans_start + " " + ", ".join(seg_token_parts) + ans_end
                # )
                #使用多个seg
                # answers.append(random.choice(self.answer_list))

                #使用多个seg+non
                # answers.append(
                #     ans_start + " " + ", ".join(seg_token_parts) + ans_end + "其他:  [NON]"
                # )
                #使用多个seg和多个non
                answers.append(
                    ans_start + " " + ", ".join(seg_token_parts) + ans_end + " " + ", ".join(non_token_parts)
                )


                # conv.append_message(conv.roles[1],answers[0])
                # conv.append_message(conv.roles[1], "[SEG].")
            else:
                text = ",".join(pest_list)
                # conv.append_message(
                #     conv.roles[0],
                #     DEFAULT_IMAGE_TOKEN
                #     + "\n  {}在图片中的什么地方? 请给出分割掩码。".format(
                #         text
                #     ),
                # )
                answers = []
                ans_start = random.choice(ANSWER_LIST_MODE4_START_PEST)
                ans_end = random.choice(ANSWER_LIST_MODE4_END)
                seg_token_parts = []
                non_token_parts = []
                for pest_name in pest_list:
                    question_template = random.choice(self.class_name_answer)
                    question_template = question_template.format(class_name=pest_name)
                    seg_token_parts.append(question_template)

                for non_pest_name in non_pest_list:
                    ans_template = random.choice(self.class_name_answer_non)
                    ans_template = ans_template.format(class_name=non_pest_name)
                    non_token_parts.append(ans_template)
                #使用多个seg
                # answers.append(
                #     ans_start + " " + ", ".join(seg_token_parts) + ans_end
                # )
                #使用单个seg
                # answers.append(random.choice(self.answer_list))
                #使用多个seg+non
                # answers.append(
                #     ans_start + " " + ", ".join(seg_token_parts) + ans_end + "其他: [NON]"
                # )

                #使用多个seg和多个non
                answers.append(
                    ans_start + " " + ", ".join(seg_token_parts) + ans_end + " " + ", ".join(non_token_parts)
                )

                # conv.append_message(conv.roles[1],answers[0])
                # conv.append_message(conv.roles[1], "[SEG]。")
            # conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = []

        # preprocess image for sam
        # image = self.transform.apply_image(image)
        # resize = image.shape[:2]
        # image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = mask_json

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        if len(non_merged_mask_) != 0:
            non_merged_mask = non_merged_mask_
            non_merged_mask = np.stack(non_merged_mask, axis=0)
            non_merged_mask = torch.from_numpy(non_merged_mask)
        else:
            non_merged_mask = torch.zeros_like(masks)
            non_merged_mask = non_merged_mask[:0]
        inference = True
        print(conversations)
        return (
            image_path,
            None,
            image_clip,
            conversations,
            masks,
            labels,
            None,
            sampled_sents,
            None,
            inference,
            pest_list,
            bbox_list,
            all_bbox_list,
            all_pest_list,
            all_mask_list,
            non_merged_mask
        )

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif isinstance(v, list) and len(v) > 0:
            input_dict[k] = [ele.cuda(non_blocking=True) if isinstance(ele, torch.Tensor) else ele for ele in v]
    return input_dict

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    # Remove or comment out the all_reduce method
    # def all_reduce(self):
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     if isinstance(self.sum, np.ndarray):
    #         total = torch.tensor(
    #             self.sum.tolist()
    #             + [
    #                 self.count,
    #             ],
    #             dtype=torch.float32,
    #             device=device,
    #         )
    #     else:
    #         total = torch.tensor(
    #             [self.sum, self.count], dtype=torch.float32, device=device
    #         )

    #     dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    #     if total.shape[0] > 2:
    #         self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
    #     else:
    #         self.sum, self.count = total.tolist()
    #     self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)

def collate_fn_val(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    text_list = []
    bboxes_list = []
    all_bboxes_list = []
    all_pests_list = []
    all_masks_list = []
    non_masks_lists = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
        pest_list,
        bbox_list,
        all_bbox_list,
        all_pest_list,
        all_mask_list,
        non_masks_list
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        text_list.append(pest_list)
        bboxes_list.append(bbox_list)
        all_bboxes_list.append(all_bbox_list)
        all_pests_list.append(all_pest_list)
        all_masks_list.append(all_mask_list)
        non_masks_lists.append(non_masks_list)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = []


    return {
        "image_paths": image_path_list,
        "images": [],
        "images_clip": "torch.stack(images_clip_list, dim=0)",
        "input_ids": input_ids,
        "labels": "targets",
        "attention_masks": "attention_masks",
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "text_list": text_list,
        "bbox_list": bboxes_list,
        "all_bbox_list": all_bboxes_list,
        "all_pest_list": all_pests_list,
        "all_mask_list": all_masks_list,
        "non_masks_list": non_masks_lists
    }

def get_pest_mask(segment_list):
    # 获取掩码图
    # 初始化一个空的变量用于存储合并结果
    merged_mask = None
    # 遍历所有的掩码图并合并
    for mask_file in segment_list:
        if "\\" in mask_file:
            mask_basename = mask_file.split('\\')[-1]  # 获取最后一个元素
        elif "/" in mask_file:
            mask_basename = mask_file.split('/')[-1]  # 获取最后一个元素
        else:
            mask_basename = mask_file
        mask_root_path = "/home/luohuibin/pycharm_workspace/SAM2/pest24_data/Pest24/mask_image/"
        mask_file = os.path.join(mask_root_path, mask_basename)
        # 读取掩码图像
        mask_image = Image.open(mask_file)
        mask_array = np.array(mask_image)
        # 将掩码图像从255/0转换为1/0
        mask_array = (mask_array > 0).astype(np.uint8)  # 大于0的值变为1，0的值保持为0
        # 如果这是第一个掩码图，直接初始化merged_mask
        if merged_mask is None:
            merged_mask = mask_array
        else:
            # 对所有掩码图进行按位或运算合并
            merged_mask = np.bitwise_or(merged_mask, mask_array)
    # 确保掩码是uint8格式
    mask = merged_mask.astype(np.int32)
    return mask

def get_mask_from_json_pest_mul(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    pest_name_list = anno["ann"]
    # pest_list = []
    # for pest_name in pest_name_list:
    #     pest = pest_name["pest_name"]
    #     pest_list.append(pest)
    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]
    annotations = anno['ann']
    segment_list = []
    pest_list = []
    bbox_list = []

    non_segment_list = []
    non_pest_list = []
    non_bbox_list = []
    for annotation in annotations:
        if "is_target" in annotation:
            segment = annotation["segmentation"]
            pest_name = annotation["pest_name"]
            bbox = annotation["bbox"]
            if annotation["is_target"]:
                mask = get_pest_mask(segment)
                segment_list.append(mask)
                pest_list.append(pest_name)
                bbox_list.append(bbox)
            else:
                mask = get_pest_mask(segment)
                non_segment_list.append(mask)
                non_pest_list.append(pest_name)
                non_bbox_list.append(bbox)
        else:
            segment = annotation["segmentation"]
            pest_name = annotation["pest_name"]
            bbox = annotation["bbox"]
            mask = get_pest_mask(segment)
            segment_list.append(mask)
            pest_list.append(pest_name)
            bbox_list.append(bbox)

    return segment_list, comments, is_sentence, pest_list, bbox_list,non_segment_list, non_pest_list, non_bbox_list


def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_pattern = r'{[^}]+}'  # 匹配最简单的JSON对象
    json_match = re.search(json_pattern, output_text)
    # pdb.set_trace()
    if json_match:
        data = json.loads(json_match.group(0))
        # 查找bbox键
        bbox_key = next((key for key in data.keys() if 'bbox' in key.lower()), None)
        # pdb.set_trace()
        if bbox_key and len(data[bbox_key]) == 4:
            content_bbox = data[bbox_key]
            content_bbox = [round(int(content_bbox[0]) * x_factor), round(int(content_bbox[1]) * y_factor),
                            round(int(content_bbox[2]) * x_factor), round(int(content_bbox[3]) * y_factor)]
        # 查找points键
        points_keys = [key for key in data.keys() if 'points' in key.lower()][:2]  # 获取前两个points键
        if len(points_keys) == 2:
            point1 = data[points_keys[0]]
            point2 = data[points_keys[1]]
            point1 = [round(int(point1[0]) * x_factor), round(int(point1[1]) * y_factor)]
            point2 = [round(int(point2[0]) * x_factor), round(int(point2[1]) * y_factor)]
            points = [point1, point2]

    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    if think_match:
        think_text = think_match.group(1)

    return content_bbox, points, think_text

import numpy as np
from PIL import Image
import torch


def get_output(text, img_path, reasoning_model, predictor, processor, h=24, w=24):
    import debugpy
    debugpy.listen(("127.0.0.1", 5678))
    print("✅ Debugpy listening on port 5678... Attach from VSCode now!")
    debugpy.wait_for_client()
    """
    text: referring instruction
    img_path: path to image
    reasoning_model: LLaVA model loaded with load_pretrained_model
    predictor: SamPredictor (SAM Vit-H)
    processor: tokenizer & image processor
    h, w: mask resolution from model path (p16 -> 16x16, p24 -> 24x24)
    """

    # 1. 读取图像并 resize 成 LLaVA 输入
    image = Image.open(img_path).convert("RGB") # <PIL.Image.Image image mode=RGB size=800x600 at 0x7EFBC3CCF8B0>
    new_w, new_h = 336, 336
    image_new = image.resize((new_w, new_h), Image.BILINEAR) # <PIL.Image.Image image mode=RGB size=336x336 at 0x7EFBC3CCE440>

    # 2. 构造 prompt（和 model_refer_seg.py 完全一致）
    text_clean = text.replace(",", "") 
    qs = random.choice(QUESTION_PARTIAL).replace("[class_name]", text_clean) # "Where is '棉铃虫' in this image? Please output segmentation mask."

    if reasoning_model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    # qs "<image>\nWhere is '棉铃虫' in this image? Please output segmentation mask."
    conv = conv_templates["llava_v1"].copy() # Conversation(system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.", roles=('USER', 'ASSISTANT'), messages=[], offset=0, sep_style=<SeparatorStyle.TWO: 2>, sep=' ', sep2='</s>', version='v1', skip_next=False)
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() # "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhere is '棉铃虫' in this image? Please output segmentation mask. ASSISTANT:"

    # 3. 图像预处理（LLaVA CLIP-ViT）
    image_tensor = process_images([image_new], processor.image_processor, reasoning_model.config)[0] # torch.Size([3, 336, 336])
    input_ids = tokenizer_image_token(prompt, processor.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') # torch.Size([66])

    # 4. LLaVA 生成 coarse mask 序列
    with torch.inference_mode():
        output_ids = reasoning_model.generate(
            input_ids.unsqueeze(0).cuda(),
            images=[image_tensor.unsqueeze(0).half().cuda()],
            image_sizes=[(new_h, new_w)],
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=256,
        )

    # 5. 解码 <seg>...</seg>
    outputs = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip() # "Sure, Here's the segmentation of the 棉铃虫:\n<seg>棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n</seg>"

    try:
        mask_labels = outputs.split("<seg>")[1].split("</seg>")[0] # '棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n 棉铃虫 *16\n'
        mask_labels = decode_mask(mask_labels)  # '棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫|棉铃虫'
        pred_mask = translate_sequence(mask_labels)
    except:
        # generate failure → zero mask
        pred_mask = [0] * (h * w) # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]

    # 6. 修正序列长度
    if len(pred_mask) < h * w:
        pred_mask = pred_mask + [pred_mask[-1]] * (h * w - len(pred_mask))
    else:
        pred_mask = pred_mask[:h * w] # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]

    pred_mask = torch.tensor(pred_mask).reshape(h, w) # torch.Size([24, 24])

    # 二值化
    pred_mask = (pred_mask > 0).int() # torch.Size([24, 24])

    # 7. 上采样 coarse mask → 原图大小
    H, W = image.size[1], image.size[0] # (600, 800)
    coarse_mask = F.interpolate(pred_mask[None, None].double(), size=(H, W), mode='nearest')[0,0] # torch.Size([600, 800])

    # 8. SAM refine
    predictor.set_image(np.array(image)) # <llava.model.segment_anything.predictor.SamPredictor object at 0x7efa47f43f40>

    if 1 not in pred_mask:
        # no positive region
        final_mask = np.zeros((H, W), dtype=np.uint8) # (600, 800)
    else:
        # coarse logits
        logits = compute_logits_from_mask(coarse_mask.double())

        # point sampling from coarse mask
        point_coords, point_labels = masks_sample_points(coarse_mask)

        sam_mask, score, logit = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=logits,
            multimask_output=False,
        )

        # 多次 refinement
        for _ in range(2):
            sam_mask, score, logit = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=logit,
                multimask_output=False,
            )

        final_mask = sam_mask[0].astype("uint8")

    final_mask = torch.tensor(final_mask, dtype=torch.uint8).cuda()   # 如果模型在 GPU torch.Size([600, 800])
    return final_mask




def validate(val_loader, epoch, writer,reasoning_model,segmentation_model,processor,test_type="refer"):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)


    pest24_dict = {'稻飞虱': '1', '稻纵卷叶螟': '2', '二化螟': '3', '黏虫': '5', '棉铃虫': '6', '草地螟': '7', '二点委夜蛾': '8', '斜纹夜蛾': '10', '甜菜夜蛾': '11', '茎蛀虫?': '12', '小壁虎?': '13', '小菜蛾': '14', '': '15', '三叶草夜蛾': '16', '黄地老虎': '24', '小地老虎': '25', '八字地老虎': '28', '大黑鳃金龟': '29', '暗黑鳃金龟': '31', '铜绿丽金龟': '32', '东方蝼蛄': '34', '线虫': '35', '金针虫': '36', '麦蛾': '37'}
    count_pest = {}
    sum_pest = {}

    for k,y in pest24_dict.items():
        count_pest[f"count_pest_{y}"] = AverageMeter(f"count_pest_{y}", ":6.3f", Summary.SUM)
        sum_pest[f"sum_pest_{y}"] = AverageMeter(f"sum_pest_{y}", ":6.3f", Summary.SUM)
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)

        input_dict["images"] = input_dict["images"]
        # input_dict["images_clip"] = input_dict["images_clip"].bfloat16()


        questions_list = input_dict['questions_list'][0][0].replace("根据图片回答,", "").strip()
        pest_tex = ','.join(input_dict["text_list"][0])
        img_path = input_dict['image_paths'][0]

        if test_type == "refer":
            query = pest_tex
        elif test_type == "reason":
            query = questions_list
        masks_list = input_dict['masks_list'][0].int()
        try:
            output_masks = get_output(query, img_path,reasoning_model,segmentation_model,processor)
            # output_masks = get_output(
            #     query,
            #     img_path,
            #     reasoning_model,
            #     segmentation_model,
            #     tokenizer,
            #     image_processor
            # )
        except Exception as e:
            print(f"[Test Catch] Exception caught: {e}")
            height, width = masks_list.shape[-2], masks_list.shape[-1]
            output_masks = torch.zeros((height, width), dtype=torch.int)

        all_pests_list = input_dict["all_pest_list"]
        all_bboxes_list = input_dict["all_bbox_list"]
        all_masks_list = input_dict["all_mask_list"]
        bboxes_list = input_dict["bbox_list"]

        pred_masks = output_masks.unsqueeze(0)



        # output_list = (pred_masks[0] > 0).int()
        # 假设原始张量形状是 (600, 800)
        # tensor = torch.randn(2, 600, 800)
        # output_list = torch.max(output_list,dim=0)[0].unsqueeze(0)
        # tensor = torch.max(tensor, dim=0)[0].unsqueeze(0)




        assert len(pred_masks) == 1

        def get_com_mask(masks_list):
            # 合并后的掩码图：按位“或”操作
            merged_mask = torch.max(masks_list, dim=0)[0]
            merged_mask = merged_mask.unsqueeze(0).to(masks_list.device)
            return merged_mask

        if masks_list.size(0) != 1:
            merged_mask = get_com_mask(masks_list)
            masks_list = merged_mask

        output_list = pred_masks.to(masks_list.device)
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        # bbox_acc_iou = []  # 用于存储每个样本的 bbox IoU 列表
        for mask_i, output_i, bbox_list,all_bbox_list,all_pest_list, all_mask_list in zip(masks_list, output_list, bboxes_list,all_bboxes_list,all_pests_list, all_masks_list):
            tt_list = []
            for m in all_mask_list:
                m = torch.tensor(m, dtype=torch.int32)
                m = m.to(output_i.device)
                tt_list.append(m)
            all_mask_list = tt_list
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target


            # 新增：基于 bbox 的 IoU 计算
            #设置一个label表明那些害虫iou应该不存在，哪些应该是0。
            # 生成标签列表
            labels = []
            for i, item in enumerate(all_pest_list):
                label = 1 if item in input_dict["text_list"][0] else 0
                # 如果 all_bbox_list[i] 的长度大于 1，将标签扩展为一个列表
                if len(all_bbox_list[i]) > 1:
                    labels.append([label] * len(all_bbox_list[i]))
                else:
                    labels.append([label])
            #找到目标害虫的id

            for pest_n,bboxes_s in zip(input_dict["text_list"][0],bboxes_list[0]):
                pest_id = pest24_dict[pest_n]
                sum_pest_id = f"sum_pest_{pest_id}"
                p_num = len(bboxes_s)
                if sum_pest_id in sum_pest:
                    sum_pest[sum_pest_id].update(1,n=p_num)
                    # pest_id_gt = pest_id
                # # 更新计数器和总和
                # if pest_id == "6":
                #     sum_pest_6 += 1
                #     pest_id_gt = pest_id
                # elif pest_id == "32":
                #     sum_pest_32 += 1
                #     pest_id_gt = pest_id
                # elif pest_id == "34":
                #     sum_pest_34 += 1
                #     pest_id_gt = pest_id
            bbox_iou_list = []
            is_more_pred = False
            for i, label_list in enumerate(labels):
                bboxes = all_bbox_list[i]
                pest_name = all_pest_list[i]
                mask = all_mask_list[i]
                # mask = torch.tensor(mask, dtype=torch.int32)
                # mask = mask.to(output_i.device)
                bbox_iou_list_tmp = []
                for lbl,bbox in zip(label_list,bboxes):
                    x_min, y_min, x_max, y_max = bbox  # 假设 bbox 是 [x_min, y_min, x_max, y_max]
                    # 确保 bbox 在 mask_i 和 output_i 的范围内
                    x_min, x_max = max(0, x_min), min(output_i.shape[1], x_max)
                    y_min, y_max = max(0, y_min), min(output_i.shape[0], y_max)
                    # 裁剪 mask 和 output
                    mask_crop = mask[y_min:y_max, x_min:x_max]
                    output_crop = output_i[y_min:y_max, x_min:x_max]
                    # 计算 bbox 内的 IoU
                    bbox_intersection, bbox_union, _ = intersectionAndUnionGPU(
                        output_crop.contiguous().clone(),
                        mask_crop.contiguous(),
                        2,
                        ignore_index=255
                    )
                    # 计算并存储当前 bbox 的 IoU
                    bbox_iou = bbox_intersection / (bbox_union + 1e-5)
                    bbox_iou = bbox_iou.cpu().numpy()
                    bbox_iou_list_tmp.append(bbox_iou[1])
                    if lbl == 0:
                        # if bbox_union[1] > 100:
                        if bbox_iou[1] > 0.2:
                            # # 计算多掩码的地方占bbox的比例
                            # total_mask_crop_pixels = mask_crop.numel()
                            # total_mask_crop_pixels = torch.tensor(
                            #     total_mask_crop_pixels, device='cpu', dtype=torch.int32
                            # )
                            # target_rate = bbox_union[1] / total_mask_crop_pixels
                            # target_rate = target_rate.cpu().numpy()
                            is_more_pred = True
                bbox_iou_list.append(bbox_iou_list_tmp)
            if not is_more_pred:
                for i, label_list in enumerate(labels):
                    bboxes = all_bbox_list[i]
                    pest_name = all_pest_list[i]
                    mask = all_mask_list[i]
                    pest_id = pest24_dict[pest_name]
                    for j, lbl in enumerate(label_list):
                        iou = bbox_iou_list[i][j]
                        if lbl == 1:
                            # assert all_mask_list[i].equal(mask_i)
                            if iou > 0.5:
                                # 更新计数器和总和
                                count_pest_id = f"count_pest_{pest_id}"
                                count_pest[count_pest_id].update(1,n=1)
                                # if pest_id_gt == "6":
                                #     count_pest_6 += 1
                                # elif pest_id_gt == "32":
                                #     count_pest_32 += 1
                                # elif pest_id_gt == "34":
                                #     count_pest_34 += 1

        # print("count_pest:",count_pest)

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]


        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    # intersection_meter.all_reduce()
    # union_meter.all_reduce()
    # acc_iou_meter.all_reduce()


    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    # for k1,y1 in pest24_dict.items():
    #     count_pest[f"count_pest_{y1}"].all_reduce()
    #     sum_pest[f"sum_pest_{y1}"].all_reduce()

    writer.add_scalar("val/giou", giou, epoch)
    writer.add_scalar("val/ciou", ciou, epoch)
    print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
    # print("count_pest_6: {:.4f}, sum_pest_6: {:.4f}, acc_pest_6: {:.4f}".format(count_pest_6, sum_pest_6,count_pest_6/sum_pest_6))
    # print("count_pest_32: {:.4f}, sum_pest_32: {:.4f}, acc_pest_32: {:.4f}".format(count_pest_32, sum_pest_32,count_pest_32/sum_pest_32))
    # print("count_pest_34: {:.4f}, sum_pest_34: {:.4f}, acc_pest_34: {:.4f}".format(count_pest_34, sum_pest_34, count_pest_34/sum_pest_34))
    # 定义保存路径
    output_path = "results.txt"
    # 以追加模式写入文件
    with open(output_path, "a") as f:
        f.write(f"Epoch: {epoch}\n")  # 写入当前 epoch
        f.write("giou: {:.4f}, ciou: {:.4f}\n".format(giou, ciou))
        av_acc = 0
        sum_acc = 0
        sum_pest_num = 0
        for n,i in pest24_dict.items():
            if sum_pest[f"sum_pest_{i}"].count != 0:
                sum_pest_num += 1
                #####可用的2025.1.5
                # f.write("count_pest_{}: {:.4f}, sum_pest_{}: {:.4f}, acc_pest_{}: {:.4f}\n".format(i,
                #     count_pest[f"count_pest_{i}"],i, sum_pest[f"sum_pest_{i}"],i, count_pest[f"count_pest_{i}"] / sum_pest[f"sum_pest_{i}"]))
                #####
                f.write("count_pest_{}: {:.4f}, sum_pest_{}: {:.4f}, acc_pest_{}: {:.4f}\n".format(i,
                                                                                                   count_pest[
                                                                                                       f"count_pest_{i}"].count,
                                                                                                   i, sum_pest[
                                                                                                       f"sum_pest_{i}"].count,
                                                                                                   i, count_pest[
                                                                                                       f"count_pest_{i}"].count /
                                                                                                   sum_pest[
                                                                                                       f"sum_pest_{i}"].count))

                acc = count_pest[f"count_pest_{i}"].count / sum_pest[f"sum_pest_{i}"].count
                sum_acc += acc

        f.write("av_acc: {:.4f}\n".format(sum_acc / sum_pest_num))
        # f.write("count_pest_6: {:.4f}, sum_pest_6: {:.4f}, acc_pest_6: {:.4f}\n".format(
        #     count_pest_6, sum_pest_6, count_pest_6 / sum_pest_6))
        # f.write("count_pest_32: {:.4f}, sum_pest_32: {:.4f}, acc_pest_32: {:.4f}\n".format(
        #     count_pest_32, sum_pest_32, count_pest_32 / sum_pest_32))
        # f.write("count_pest_34: {:.4f}, sum_pest_34: {:.4f}, acc_pest_34: {:.4f}\n".format(
        #     count_pest_34, sum_pest_34, count_pest_34 / sum_pest_34))
        f.write("\n")  # 添加一个空行，便于区分不同 epoch 的内容
        # av_acc_ = sum_acc / sum_pest_num

    return giou, ciou

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="/mnt/data/home/lilanting/shenjie/code/Text4SegHub/checkpoints/llava-v1.5-7b-p16")
    parser.add_argument("--segmentation_model_path", type=str, default="/mnt/data/home/lilanting/shenjie/code/Text4SegHub/llava/model/segment_anything/sam_vit_h_4b8939.pth")
    parser.add_argument("--text", type=str, default="the unusal object in the image")
    parser.add_argument("--image_path", type=str, default="/mnt/data/home/luohuibin/pycharm_wordspace/PestSegVllm_comparison/Seg-Zero-main/assets/test_image.png")
    parser.add_argument("--output_path", type=str, default="./inference_scripts/test_output.png")
    return parser.parse_args()

from llava.model.segment_anything import SamPredictor, sam_model_registry
from types import SimpleNamespace

def main():

    args = parse_args()

    data_type = "refer"
    dataset_dir = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/cleaned_data/refer_data/simple_data/"
    image_root_path = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/cleaned_data/images/"
    # reason_seg_data_add = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/cleaned_data/reason_data/medium_data"
    # reason_image_root_path = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/cleaned_data/images/"
    reason_seg_data_add = None
    reason_image_root_path = None

    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     args.reasoning_model_path,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )

    # segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)
    model_name = get_model_name_from_path(args.reasoning_model_path)
    tokenizer, reasoning_model, image_processor, context_len = load_pretrained_model(
        args.reasoning_model_path,
        None,
        model_name
    )
    sam = sam_model_registry["vit_h"](checkpoint=args.segmentation_model_path)
    sam = sam.to(dtype=torch.float32, device='cuda')
    segmentation_model = SamPredictor(sam)

    reasoning_model.eval()

    # default processer
    # processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    processor = SimpleNamespace()
    processor.image_processor = image_processor
    processor.tokenizer = tokenizer

    query = args.text
    img_path  = args.image_path

    val_dataset = ValDatasetPest(
        dataset_dir,
        "/mnt/data/home/luohuibin/Model/clip-vit-large-path14/",
        "ReasonSeg|val",
        1024,
        image_root_path=image_root_path,
        reason_seg_data_add=reason_seg_data_add,
        reason_image_root_path=reason_image_root_path
    )

    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    #     val_dataset, shuffle=False, drop_last=False
    # )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=partial(
            collate_fn_val,
            conv_type="llava_v1",
            use_mm_start_end=True,
            local_rank=0
        ))

    writer = SummaryWriter("test")

    validate(val_loader, 0, writer,reasoning_model,segmentation_model,processor, data_type)
    # out_mask = get_output(query,img_path,reasoning_model,segmentation_model,processor)
if __name__ == "__main__":
    # os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
