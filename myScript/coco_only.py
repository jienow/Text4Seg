import json
import os
from dotenv import load_dotenv
load_dotenv('.env')
project_root = os.getenv('PROJECT_ROOT', "")
# 输入文件名
input_file = project_root + '/playground/llava_v1_5_mix665k.json'
# 输出文件名
output_file = 'playground/coco_only.json'

def filter_coco_data():
    try:
        print(f"正在读取 {input_file} ...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 过滤逻辑：保留 image 路径中包含 'coco' 的项
        # 同时也加入了一个简单的防错机制，确保 item 包含 'image' 字段
        coco_data = [
            item for item in data 
            if 'image' in item and 'coco' in item['image'].lower()
        ]
        
        print(f"原始数据条数: {len(data)}")
        print(f"筛选后 COCO 数据条数: {len(coco_data)}")
        
        print(f"正在写入 {output_file} ...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
            
        print("完成！")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    filter_coco_data()