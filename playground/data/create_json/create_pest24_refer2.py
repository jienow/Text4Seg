import os
import json
import cv2
import numpy as np
import random
from tqdm import tqdm

# ================= 配置区域 =================

PREPROCESSED_JSON_PATH = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/pest24Refer/processed_dataset.json"
ORIGINAL_JSON_DIR = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/refer_data/dif_data/reason_seg/ReasonSeg/train/"
IMAGE_ROOT = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/refer_data/dif_data/images/"
# 假设 Mask 就在 Image 目录下，或者你需要单独定义 MASK_ROOT
MASK_ROOT = "/home/luohuibin/pycharm_workspace/SAM2/pest24_data/"

OUTPUT_FINAL_JSON = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/json_files/pest24_text4seg_train_refer_shi.json"

H_NEW = 16
W_NEW = 16

# ===================================================================

QUESTION_PARTIAL = [
    '你能在这张图片中分割出[class_name]吗?请输出分割掩码并给出害虫名。',
    '请在这张图片中分制出[class_name]。请输出分割掩码并给出害虫名。',
    '这张图片中的[class_name]在哪里?请返回分割掩码。'
]
ANSWER_PARTIAL = [
    '好的，这是‘[class_name]’的分割掩码：',
    '这是聚焦于‘[class_name]’的分割图：',
    '这是突出显示‘[class_name]’的分割掩码：'
]

def encode_mask(mask_array):
    if len(mask_array) == 0:
        return ""
    encoded = []
    current_token = mask_array[0]
    count = 1
    for token in mask_array[1:]:
        if token == current_token:
            count += 1
        else:
            encoded.append(f"{current_token}*{count}")
            current_token = token
            count = 1
    encoded.append(f"{current_token}*{count}")
    return "|".join(encoded)

def load_original_json(original_json_name):
    full_path = os.path.join(ORIGINAL_JSON_DIR, original_json_name)
    if not os.path.exists(full_path):
        return None
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            data = data[0]
        return data
    except Exception as e:
        print(f"Error reading original json {full_path}: {e}")
        return None

# 修改：增加 orig_w, orig_h 参数
def force_mask(high_res_mask, mask_resized, orig_w, orig_h):
    # 逻辑：如果高分辨图有物体(max>0)，但缩放后全黑(max==0)，则强制找回
    if high_res_mask.max() > 0 and mask_resized.max() == 0:
        M = cv2.moments(high_res_mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            target_x = int(cX * (W_NEW / orig_w))
            target_y = int(cY * (H_NEW / orig_h))
            
            target_x = min(max(target_x, 0), W_NEW - 1)
            target_y = min(max(target_y, 0), H_NEW - 1)
            
            mask_resized[target_y, target_x] = 1
    return mask_resized

def process_final_dataset():
    print(f"正在读取预处理数据: {PREPROCESSED_JSON_PATH}")
    with open(PREPROCESSED_JSON_PATH, 'r', encoding='utf-8') as f:
        preprocessed_data = json.load(f)
    
    print(f"共加载 {len(preprocessed_data)} 条数据，开始转换...")
    
    final_content = []
    
    # 限制用于测试，正式跑请去掉 [:100] 或 enumerate 中的 break
    for idx, item in tqdm(enumerate(preprocessed_data)):
        # if idx > 5000: break 
        try:
            # --- A. 获取基础信息 ---
            original_json_name = item['original_json']
            json_data = load_original_json(original_json_name)
            
            # 容错：防止读不到
            if json_data is None: continue

            # 获取原图尺寸 (为了坐标映射)
            # 优先从 JSON shape 里找，找不到再尝试读文件
            if 'shapes' in json_data and len(json_data['shapes']) > 0:
                 original_image = json_data.get('shapes')[0].get('image_name')
            else:
                 original_image = json_data.get('imagePath')

            if original_image is None:
                continue

            image_full_path = os.path.join(IMAGE_ROOT, original_image)
            
            # 读取原图尺寸
            if os.path.exists(image_full_path):
                img_cv = cv2.imread(image_full_path)
                if img_cv is None: continue
                orig_h, orig_w = img_cv.shape[:2]
            else:
                # 如果找不到图，没法算比例，跳过
                continue

            target_pest_list = item['selected_pests_detail']
            
            # 1. 初始化底板 (列表乘法)
            # [Fix]: 使用 ['others'] * 256 而不是 'others'*256
            final_label_list = ['others'] * (H_NEW * W_NEW)
            
            # 用于收集本图中出现的所有害虫名，构造 question 用
            found_pest_names = set()

            for target_pest_dir in target_pest_list:
                pest_name = target_pest_dir['pest_name']
                segmentationPathList = target_pest_dir['segmentation']
                
                # 如果没有分割路径，跳过
                if not segmentationPathList:
                    continue

                found_pest_names.add(pest_name)
                
                # 处理该害虫对应的所有 mask 文件
                for segPath in segmentationPathList:
                    mask_full_path = os.path.join(MASK_ROOT, segPath)
                    
                    if not os.path.exists(mask_full_path):
                        continue
                        
                    # [Fix]: 0 模式读取灰度图
                    high_res_mask = cv2.imread(mask_full_path, 0) 
                    if high_res_mask is None: continue
                    
                    # 二值化
                    high_res_mask[high_res_mask > 0] = 1
                    
                    # 缩放
                    mask_resized_img = cv2.resize(high_res_mask, (W_NEW, H_NEW), interpolation=cv2.INTER_NEAREST)
                    
                    # [Fix]: 传入高分辨原图 + 缩放图 + 宽高
                    mask_resized_img = force_mask(high_res_mask, mask_resized_img, orig_w, orig_h)
                    
                    mask_flatten = mask_resized_img.flatten()
                    
                    # 转换为当前 mask 的 label 列表
                    current_label_list = [pest_name if val > 0 else "others" for val in mask_flatten]
                    
                    # [Merge Logic]: 
                    # 如果当前位置是 pest_name，则覆盖 final_label_list
                    # 如果当前位置是 others，则保留 final_label_list 原有的值 (可能是之前画的其他害虫)
                    final_label_list = [
                        curr if curr != 'others' else final 
                        for curr, final in zip(current_label_list, final_label_list)
                    ]

            # 构造 class_name_str (e.g. "Bollworm and Moth")
            if not found_pest_names:
                continue # 如果这一圈下来一个害虫都没画上，跳过
                
            unique_names = sorted(list(found_pest_names))
            if len(unique_names) > 1:
                class_name_str = ", ".join(unique_names[:-1]) + " 和 " + unique_names[-1]
            else:
                class_name_str = unique_names[0]

            # RLE 编码
            seg_string = encode_mask(final_label_list)
            
            # --- F. 构造对话 ---
            q_tmpl = random.choice(QUESTION_PARTIAL)
            a_tmpl = random.choice(ANSWER_PARTIAL)
            
            question_text = q_tmpl.replace("[class_name]", class_name_str)
            answer_text_base = a_tmpl.replace("[class_name]", class_name_str)
            answer_final = f"{answer_text_base}\n<seg>{seg_string}</seg>"
            
            val_human = f"<image>\n{question_text}"
            
            final_item = {
                "id": os.path.splitext(original_json_name)[0],
                "image": image_full_path,
                "height": H_NEW,
                "width": W_NEW,
                "conversations": [
                    {"from": "human", "value": val_human},
                    {"from": "gpt", "value": answer_final}
                ]
            }
            final_content.append(final_item)
            
        except Exception as e:
            print(f"处理数据项 {item.get('original_json', 'unknown')} 时出错: {e}")
            # import traceback
            # traceback.print_exc()
            continue

    # --- G. 写入文件 ---
    print(f"处理完成，共生成 {len(final_content)} 条最终数据。")
    os.makedirs(os.path.dirname(OUTPUT_FINAL_JSON), exist_ok=True)
    
    with open(OUTPUT_FINAL_JSON, "w", encoding='utf-8') as f:
        json.dump(final_content, f, indent=4, ensure_ascii=False)
    print(f"写入完成: {OUTPUT_FINAL_JSON}")

if __name__ == "__main__":
    process_final_dataset()