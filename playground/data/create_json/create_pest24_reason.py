import os
import json
import cv2
import numpy as np
import glob
import random
import sys
from tqdm import tqdm

# ================= 配置区域 (已根据你的要求修改) =================

# 1. 原始 JSON 文件所在的目录 (里面有很多个单独的 json)
INPUT_JSON_DIR = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/reason_data/dif_data/reason_seg/ReasonSeg/train/"

# 2. 原图的根目录
# 拼接逻辑: os.path.join(DATA_ROOT, image_name)
DATA_ROOT = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/reason_data/dif_data/images/"

# 3. Mask 的根目录
# 拼接逻辑: os.path.join(MASK_ROOT, json里的segmentation路径)
MASK_ROOT = "/home/luohuibin/pycharm_workspace/SAM2/pest24_data/"

# 4. 输出文件的路径
OUTPUT_JSON_FILE = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/json_files/pest24_text4seg_train_reason.json"

# 5. Text4SegHub 代码库的基础路径
# 如果 question_answer_list.py 和此脚本在同一目录，可以不修改；否则需要指向该文件所在文件夹
# CODE_BASE_PATH = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/code_base"  # 假设这是你的代码库路径，如果不同请修改

# 6. 设置生成参数
H_NEW = 16  # 目标 Mask 高度
W_NEW = 16  # 目标 Mask 宽度
ROUNDS_PER_IMAGE = 1  # 每张图生成几轮对话

# ===================================================================

# --- 动态导入外部的 Prompt 列表 ---

# from question_answer_list import (
#     QUESTION_PARTIAL, 
#     ANSWER_PARTIAL
# )
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
    # 添加最后一个
    encoded.append(f"{current_token}*{count}")
    
    return "|".join(encoded)

def process_dataset():
    # 获取目录下所有 json 文件
    json_files = glob.glob(os.path.join(INPUT_JSON_DIR, "*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件，开始处理...")

    content = []
    
    # 使用 tqdm 显示进度条
    for idx, json_file in tqdm(enumerate(json_files)):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 兼容处理：如果是列表取第一个，如果是字典直接用
            if isinstance(data, list):
                data = data[0]
            
            # --- 1. 获取基础图片信息 ---
            if "shapes" not in data or not data["shapes"]:
                continue
                
            image_rel_path = data["shapes"][0]["image_name"]
            
            # 【修改点 1】原图路径拼接
            # 直接将 DATA_ROOT 和文件名拼接，不再添加额外的 "Pest24/images"
            image_full_path = os.path.join(DATA_ROOT, image_rel_path)
            
            # 读取原图以获取 H, W (用于创建画布)
            img_cv = cv2.imread(image_full_path)
            if img_cv is None:
                # 打印一下失败路径方便 debug
                # print(f"警告: 找不到原图 {image_full_path}，跳过该条数据。")
                continue
                
            orig_h, orig_w = img_cv.shape[:2]
            
            annotations = data.get("ann", [])
            if not annotations:
                raise ValueError(f"ID {os.path.splitext(os.path.basename(json_file))[0]} 没有标注信息")

            # --- 2. 构造输出 Item ---
            item = {
                "id": os.path.splitext(os.path.basename(json_file))[0],
                "image": image_full_path, 
                "height": H_NEW,
                "width": W_NEW,
                "conversations": []
            }

            conversation_list = []
            
            # --- 3. 生成多轮对话 ---
            
            # 随机选择一个害虫标注
            ann = next((item for item in annotations if item['is_target']), None)
            if not ann:
                raise ValueError(f"ID {os.path.splitext(os.path.basename(json_file))[0]} 没有目标标注")
                
            # 获取类别名 (作为 Prompt 的输入)
            class_name = ann.get("pest_name")
            
            # --- Mask 处理核心逻辑 ---
            # 初始化全黑 Mask
            final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            
            mask_paths = ann.get("segmentation", [])

            # === 调试代码开始 ===
            if len(mask_paths) == 0:
                print(f"DEBUG: ID {item['id']} 没有 segmentation 路径")
            
            for mask_rel_path in mask_paths:
                # 【修改点 2】Mask 路径拼接
                # 使用专门的 MASK_ROOT 拼接
                mask_full_path = os.path.join(MASK_ROOT, mask_rel_path)
                
                # 1. 打印尝试读取的路径，看看对不对
                # print(f"DEBUG: 尝试读取掩码 -> {mask_full_path}")
                
                if not os.path.exists(mask_full_path):
                    print(f"❌ 错误: 文件不存在! \n路径: {mask_full_path}")
                    continue

                # 读取 Mask (灰度模式 0)
                m = cv2.imread(mask_full_path, 0)
                if m is None:
                    print(f"❌ 错误: cv2 读取失败 (可能是格式损坏): {mask_full_path}")
                    # print(f"警告: 找不到 Mask {mask_full_path}")
                    continue

                if m.max() == 0:
                    print(f"⚠️ 警告: 这张掩码图是全黑的！")    
                
                # 确保尺寸匹配 (防止 Mask 和原图尺寸不一致报错)
                if m.shape[:2] != (orig_h, orig_w):
                    m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                # 合并 Mask (逻辑或: 只要有一个是前景就是前景)
                final_mask = np.maximum(final_mask, m)
            
            # 二值化
            final_mask[final_mask > 0] = 1
            
            # Resize 到 16x16 (注意使用 NEAREST 防止产生小数)
            mask_resized = cv2.resize(final_mask, (W_NEW, H_NEW), interpolation=cv2.INTER_NEAREST)

            # ==============================================================================
            # 【插入代码开始】方案二：强制重心找回 (防止小物体因缩放消失)
            # ==============================================================================
            # 如果原图有物体 (final_mask > 0)，但在缩放后消失了 (mask_resized == 0)
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
            # ==============================================================================
            # 【插入代码结束】
            # ==============================================================================
            
            # Flatten 展平
            mask_flatten = mask_resized.flatten()
            
            # 转换为文本标签列表: 1 -> class_name, 0 -> "others"
            label_list = [class_name if val > 0 else "others" for val in mask_flatten]
            
            # RLE 编码生成 <seg> 字符串
            seg_string = encode_mask(label_list)
            if 'others*256' in seg_string:
                print(f"DEBUG: ID {item['id']} 生成的掩码中包含全黑区域，无法编码为 RLE 格式。")
                raise ValueError(f"ID {item['id']} 生成的掩码中包含全黑区域，无法编码为 RLE 格式。")
            
            # --- 4. 构造 Prompt ---
            # 生成问题
            question_tmpl = data.get('text')[0]
            question_text = question_tmpl.replace("[class_name]", class_name)
            
            # 生成回答
            answer_tmpl = random.choice(ANSWER_PARTIAL)
            answer_text = answer_tmpl.replace("[class_name]", class_name)
            
            # 拼接最终回答
            answer_final = f"{answer_text}\n<seg>{seg_string}</seg>"
            
            val_human = f"<image>\n{question_text}"
            
            conversation_list.append({"from": "human", "value": val_human})
            conversation_list.append({"from": "gpt", "value": answer_final})
        
            item["conversations"] = conversation_list
            content.append(item)

        except Exception as e:
            print(f"处理文件 {json_file} 时发生错误: {e}")
            continue
    
    # --- 5. 写入最终 JSON ---
    print(f"处理完成，共生成 {len(content)} 条数据。正在写入 {OUTPUT_JSON_FILE}...")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_JSON_FILE), exist_ok=True)
    
    with open(OUTPUT_JSON_FILE, "w", encoding='utf-8') as f:
        json.dump(content, f, indent=4, ensure_ascii=False)
    print("写入完成！")

if __name__ == "__main__":
    process_dataset()