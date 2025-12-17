import os
import json
import cv2
import numpy as np
import random
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 预处理生成的 JSON 文件路径
PREPROCESSED_JSON_PATH = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/pest24Refer/processed_dataset.json"

# 2. 原始 JSON 所在的目录 (Labelme 格式)
ORIGINAL_JSON_DIR = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/refer_data/dif_data/reason_seg/ReasonSeg/train/"

# 3. 原图的根目录
IMAGE_ROOT = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/refer_data/dif_data/images/"

# 4. 最终输出文件的路径
OUTPUT_FINAL_JSON = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/json_files/pest24_text4seg_train_refer_separated.json"

# 5. Mask 参数
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
    """
    RLE 编码函数：将 1D 数组转换为 "token*count|token*count" 格式
    """
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
    """
    读取原始 Labelme JSON 内容
    """
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

def force_mask(final_mask, mask_resized):
    if final_mask.max() > 0 and mask_resized.max() == 0:
        # print(f"DEBUG: 检测到小物体消失，正在执行重心强制映射... ID: {item['id']}")
        
        # 1. 计算原图掩码的矩 (Moments)
        M = cv2.moments(final_mask)
        
        # 确保面积不为0 (理论上max>0面积就不为0，但为了安全加个判断)
        if M["m00"] != 0:
            # 2. 算出原图中的重心坐标 (cX, cY)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # 3. 坐标映射: 从 原图尺寸 -> 映射到 16x16
            # 公式: 目标坐标 = 原坐标 * (目标宽 / 原宽)
            target_x = int(cX * (W_NEW / orig_w))
            target_y = int(cY * (H_NEW / orig_h))
            
            # 4. 防止边缘溢出 (坐标必须在 0 到 15 之间)
            target_x = min(max(target_x, 0), W_NEW - 1)
            target_y = min(max(target_y, 0), H_NEW - 1)
            
            # 5. 强制点亮该像素
            mask_resized[target_y, target_x] = 1
        return mask_resized
    return mask_resized

def process_final_dataset():
    # 1. 读取预处理好的列表
    print(f"正在读取预处理数据: {PREPROCESSED_JSON_PATH}")
    with open(PREPROCESSED_JSON_PATH, 'r', encoding='utf-8') as f:
        preprocessed_data = json.load(f)
    
    print(f"共加载 {len(preprocessed_data)} 条数据，开始转换 (多类别分离模式)...")
    
    final_content = []
    
    for idx, item in tqdm(enumerate(preprocessed_data)):
        if idx > 100:
            break
        try:
            # --- A. 获取基础信息 ---
            original_json_name = item['original_json']
            json_data = load_original_json(original_json_name)
            original_image = json_data.get('shapes')[0].get('image_name')
            if original_image is None:
                continue
            image_full_path = os.path.join(IMAGE_ROOT, original_image)
            img_cv = cv2.imread(image_full_path)
            orig_h, orig_w, _ = img_cv.shape

            target_pest_list = item['selected_pests_detail']
            final_label_list = ['others'*256]
            for target_pest_dir in target_pest_list:
                class_name_str = target_pest_dir['pest_name']
                segmentationPathList = target_pest_dir['segmentation']
                for segPath in segmentationPathList:
                    mask_full_path = os.path.join(MASK_ROOT, segPath)
                    mask = cv2.imread(mask_full_path)
                    mask[mask > 0] = 1
                    mask = cv2.resize(mask, (W_NEW, H_NEW), interpolation=cv2.INTER_NEAREST)
                    mask_resized = force_mask(final_mask=mask, mask_resized=mask)
                    mask_flatten = mask_resized.flatten()
                    # 转换为文本标签列表: 1 -> class_name, 0 -> "others"
                    label_list = [class_name_str if val > 0 else "others" for val in mask_flatten]
                    final_label_list = [item 
                        if item != 'others' else final_label_list[i] 
                        for i, item in enumerate(label_list)
                    ]
            

            # RLE 编码 (现在 label_list 包含了分开的害虫名)
            # 结果示例: others*50|Bollworm*3|others*10|RiceLeafRoller*2
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
            import traceback
            traceback.print_exc()
            continue

    # --- G. 写入文件 ---
    print(f"处理完成，共生成 {len(final_content)} 条最终数据。")
    os.makedirs(os.path.dirname(OUTPUT_FINAL_JSON), exist_ok=True)
    
    with open(OUTPUT_FINAL_JSON, "w", encoding='utf-8') as f:
        json.dump(final_content, f, indent=4, ensure_ascii=False)
    
    print(f"写入完成: {OUTPUT_FINAL_JSON}")

if __name__ == "__main__":
    process_final_dataset()