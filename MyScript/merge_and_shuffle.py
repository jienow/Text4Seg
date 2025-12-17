import json
import random
import os

def combine_and_shuffle_jsons(file_path1, file_path2, output_path):
    """
    读取两个JSON文件，分别打乱后，按顺序合并并保存。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(file_path1):
        print(f"❌ 错误: 找不到文件 1: {file_path1}")
        return
    if not os.path.exists(file_path2):
        print(f"❌ 错误: 找不到文件 2: {file_path2}")
        return

    # 2. 读取文件
    print(f"正在读取文件 1: {file_path1} ...")
    with open(file_path1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    print(f"正在读取文件 2: {file_path2} ...")
    with open(file_path2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    # 3. 验证数据格式 (必须是列表)
    if not isinstance(data1, list) or not isinstance(data2, list):
        print("❌ 错误: 两个 JSON 文件必须都是数组（List）格式。")
        return

    print(f"文件 1 包含 {len(data1)} 条数据")
    print(f"文件 2 包含 {len(data2)} 条数据")

    # 4. 在合并前分别随机打乱
    print("正在分别随机打乱两个列表...")
    random.shuffle(data1)
    random.shuffle(data2)

    # 5. 按照顺序合并 (先 List1，后 List2)
    combined_data = data1 + data2
    print(f"合并完成，总共 {len(combined_data)} 条数据")

    # 6. 保存结果
    print(f"正在写入输出文件: {output_path} ...")
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)
    
    print("✅ 处理完成！")

# ================= 配置区域 =================

# 第一个 JSON 文件路径 (排在前面)
JSON_FILE_1 = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/json_files/pest24_text4seg_train_reason.json"

# 第二个 JSON 文件路径 (排在后面)
JSON_FILE_2 = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/json_files/pest24_text4seg_train_refer.json"

# 输出合并后的 JSON 文件路径
OUTPUT_FILE = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/json_files/pest24_text4seg_train_combined.json"

# ===========================================

if __name__ == "__main__":
    combine_and_shuffle_jsons(JSON_FILE_1, JSON_FILE_2, OUTPUT_FILE)