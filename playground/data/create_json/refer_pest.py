import json
import os
import random
import cv2
import numpy as np
import glob
from tqdm import tqdm

# --- 配置区域 (请根据你的环境修改) ---
INPUT_JSON_DIR = "/mnt/data/home/luohuibin/lisa_chechpoint/PestSegVllm_data/refer_data/dif_data/reason_seg/ReasonSeg/train/"      # 原始JSON文件夹路径
MASK_SOURCE_ROOT = "/home/luohuibin/pycharm_workspace/SAM2/pest24_data/"       # 原始掩码图的根目录 (例如 Pest24 上级目录)
OUTPUT_DIR = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/pest24Refer"                   # 输出结果的文件夹
OUTPUT_MASK_DIR = os.path.join(OUTPUT_DIR, "masks") # 生成的新掩码存放路径
OUTPUT_JSON_NAME = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/pest24Refer/processed_dataset.json"     # 输出的新JSON文件名
# ------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge_masks(mask_paths, source_root):
    """
    读取多个掩码路径并将它们合并为一张二值图
    """
    final_mask = None
    
    for relative_path in mask_paths:
        full_path = os.path.join(source_root, relative_path)
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            print(f"[Warning] Mask file not found: {full_path}")
            continue
            
        # 读取掩码 (以灰度模式读取)
        mask = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"[Warning] Failed to load image: {full_path}")
            continue
            
        # 初始化画布 (第一次读取时)
        if final_mask is None:
            final_mask = np.zeros_like(mask)
            
        # 合并掩码 (逻辑或操作，只要有一个是前景，结果就是前景)
        # 假设掩码中 0 是背景，非 0 是前景
        final_mask = cv2.bitwise_or(final_mask, mask)
        
    return final_mask

def process_files(file_list, target_count, all_results):
    """
    处理文件列表，从每个文件中随机抽取 target_count 个害虫
    """
    for json_file in tqdm(file_list, desc=f"Processing group (Target {target_count} pests)"):
        filename = os.path.basename(json_file)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        ann_list = data.get('ann', [])
        
        # 如果没有害虫信息，跳过
        if not ann_list:
            continue
            
        # 随机采样逻辑
        # 如果可用害虫数量少于目标数量，则取所有可用害虫
        num_to_sample = min(len(ann_list), target_count)
        selected_pests = random.sample(ann_list, num_to_sample)
        
        # 收集所有选中害虫的所有掩码路径
        all_mask_paths = []
        en_names = []
        
        for pest in selected_pests:
            # 收集分割路径
            if 'segmentation' in pest:
                all_mask_paths.extend(pest['segmentation'])
            # 收集英文名 (用于问题来源)
            if 'pest_name' in pest:
                en_names.append(pest['pest_name'])
        
        # 生成并保存合并后的掩码
        if all_mask_paths:
            combined_mask = merge_masks(all_mask_paths, MASK_SOURCE_ROOT)
            
            if combined_mask is not None:
                # 构造新的掩码文件名
                # 命名格式: mask_原JSON名_pests数.png
                base_name = os.path.splitext(filename)[0]
                new_mask_name = f"mask_{base_name}_{num_to_sample}_pests.png"
                save_path = os.path.join(OUTPUT_MASK_DIR, new_mask_name)
                
                # 保存图片
                cv2.imwrite(save_path, combined_mask)
                
                # 构建输出数据条目
                entry = {
                    "original_json": filename,
                    "target_pest_count_group": target_count, # 所属的分组(1, 2, or 3)
                    "actual_pest_count": num_to_sample,      # 实际取到的数量
                    "question_sources": en_names,            # 英文名列表，用于提问
                    "generated_mask_path": save_path,        # 生成的合并掩码路径
                    "selected_pests_detail": selected_pests  # 选中害虫的完整原始信息
                }
                all_results.append(entry)

def main():
    # 1. 准备目录
    ensure_dir(OUTPUT_DIR)
    ensure_dir(OUTPUT_MASK_DIR)
    
    # 2. 读取所有 JSON 文件
    json_files = glob.glob(os.path.join(INPUT_JSON_DIR, "*.json"))
    
    if not json_files:
        print("No JSON files found in input directory.")
        return
        
    print(f"Found {len(json_files)} JSON files.")
    
    # 3. 随机打乱并平均分成三份
    random.shuffle(json_files)
    
    # 计算切分点
    n = len(json_files)
    chunk_size = n // 3
    
    # 处理余数，余数分配给最后一份
    split_1 = json_files[:chunk_size]
    split_2 = json_files[chunk_size:2*chunk_size]
    split_3 = json_files[2*chunk_size:]
    
    all_results = []
    
    # 4. 分别处理三组数据
    # 第一组：取 1 个害虫
    process_files(split_1, 1, all_results)
    
    # 第二组：取 2 个害虫
    process_files(split_2, 2, all_results)
    
    # 第三组：取 3 个害虫
    process_files(split_3, 3, all_results)
    
    # 5. 保存新的 JSON 文件
    output_json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON_NAME)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
        
    print(f"\nProcessing complete!")
    print(f"Total entries generated: {len(all_results)}")
    print(f"New JSON saved to: {output_json_path}")
    print(f"Masks saved to: {OUTPUT_MASK_DIR}")

if __name__ == "__main__":
    main()