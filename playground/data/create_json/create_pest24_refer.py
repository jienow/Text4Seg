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
            # 这里我们不再使用 generated_mask_path，而是重新画 Mask
            target_pest_list = item['question_sources']  # 用户问及的害虫列表 e.g. ['Bollworm', 'RiceLeafRoller']
            
            if not target_pest_list:
                continue

            # 构造Prompt中使用的文本 (e.g., "Bollworm and Rice Leaf Roller")
            unique_names = sorted(list(set(target_pest_list)))

            # 对每个 pest 类别单独处理
            for class_name_str in unique_names:
                # if len(unique_names) > 1:
                #     class_name_str = ", ".join(unique_names[:-1]) + " and " + unique_names[-1]
                # else:
                #     class_name_str = unique_names[0]
                
                # --- B. 读取原始 JSON 数据 ---
                json_data = load_original_json(original_json_name)
                
                # 获取图片路径
                image_rel_name = json_data.get("imagePath", "") # Labelme 通常是 imagePath

                image_full_path = os.path.join(IMAGE_ROOT, image_rel_name)


                # 读取原图获取尺寸
                img_cv = cv2.imread(image_full_path)
                orig_h, orig_w = img_cv.shape[:2]

                # --- C & D & E. 逐个类别处理、Resize、找回重心并合并 ---

                # 1. 初始化最终的标签列表 (长度 256，默认全是 others)
                # 这里的逻辑是：先填满 others，然后谁有掩码谁就去占坑
                total_pixels = H_NEW * W_NEW
                final_token_list = ["others"] * total_pixels

                # 2. 获取所有形状 (一次性读取，避免重复循环)
                shapes = json_data.get("shapes", [])
                
                # 3. 遍历每一个目标害虫 (按照列表顺序，排在前面的优先占坑)
                for pest_name in unique_names:
                    
                    # --- Step 1: 绘制当前害虫的二值 Mask ---
                    # 创建全黑 Mask
                    single_pest_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    
                    # 寻找匹配的 polygon
                    found_match = False
                    for shape in shapes:
                        raw_label = str(shape.get("label", ""))
                        # 简单清洗匹配：去除空格，转小写
                        if raw_label.strip().lower() == pest_name.strip().lower():
                            points = shape.get("points", [])
                            if len(points) > 0:
                                pts = np.array(points, dtype=np.int32)
                                cv2.fillPoly(single_pest_mask, [pts], 1) # 填充为 1
                                found_match = True
                    
                    # 如果原图中根本没找到这个害虫的标签，跳过处理
                    if not found_match:
                        continue

                    # --- Step 2: Resize 到 16x16 ---
                    mask_resized = cv2.resize(single_pest_mask, (W_NEW, H_NEW), interpolation=cv2.INTER_NEAREST)
                    
                    # --- Step 3: 重心找回 (Small Object Recovery) ---
                    # 如果原图有画 (found_match=True) 但是缩放后全是 0
                    if np.max(mask_resized) == 0:
                        M = cv2.moments(single_pest_mask)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            
                            target_x = int(cX * (W_NEW / orig_w))
                            target_y = int(cY * (H_NEW / orig_h))
                            
                            # 坐标截断防止越界
                            target_x = min(max(target_x, 0), W_NEW - 1)
                            target_y = min(max(target_y, 0), H_NEW - 1)
                            
                            # 强制点亮该点
                            mask_resized[target_y, target_x] = 1

                    # --- Step 4: 合并到最终结果 (Conflict Resolution: First Come First Serve) ---
                    mask_flatten = mask_resized.flatten()
                    
                    for idx, val in enumerate(mask_flatten):
                        if val > 0: # 如果在这个位置，当前害虫有像素
                            # 只有当这个坑还是 'others' 时才填入
                            # 这就是 "有冲突取第一个" 的逻辑
                            if final_token_list[idx] == "others":
                                final_token_list[idx] = pest_name

                # --- F. 编码 ---
                # 此时 final_token_list 已经变成了 ['others', 'Bollworm', 'others', 'Rice...', ...]
                seg_string = encode_mask(final_token_list)
            

            # RLE 编码 (现在 label_list 包含了分开的害虫名)
            # 结果示例: others*50|Bollworm*3|others*10|RiceLeafRoller*2
            seg_string = encode_mask(label_list)
            
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