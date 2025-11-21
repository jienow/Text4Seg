import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

# === 配置路径 (请确认你的路径是 coco_stuff 还是 cocostuff) ===
DATA_ROOT = "playground/data/coco_stuff/annotations" 

TRAIN_DIR = os.path.join(DATA_ROOT, "train2017")
VAL_DIR = os.path.join(DATA_ROOT, "val2017")
def get_label_mapping():
    # 创建一个 256 长度的数组，默认填充 255 (ignore_index)
    # 这样无论图片里出现什么奇奇怪怪的像素值 (0-255)，都不会报错
    mapping = np.ones(256, dtype=np.uint8) * 255
    
    # COCO-Stuff 164K 完整映射表 (Index 0-181 -> TrainID 0-170)
    # 原始 ID 0 是 unlabeled，映射为 255
    # 原始 ID 1-181 映射为 0-170 (部分 ID 是跳过的或合并的)
    
    # 定义有效的映射关系 (Format: Original_ID: Train_ID)
    # 这是一个标准的 COCO-Stuff 转换逻辑
    valid_mapping = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 
        11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 
        21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26, 28: 27, 29: 28, 30: 29, 
        31: 30, 32: 31, 33: 32, 34: 33, 35: 34, 36: 35, 37: 36, 38: 37, 39: 38, 40: 39, 
        41: 40, 42: 41, 43: 42, 44: 43, 45: 44, 46: 45, 47: 46, 48: 47, 49: 48, 50: 49, 
        51: 50, 52: 51, 53: 52, 54: 53, 55: 54, 56: 55, 57: 56, 58: 57, 59: 58, 60: 59, 
        61: 60, 62: 61, 63: 62, 64: 63, 65: 64, 66: 65, 67: 66, 68: 67, 69: 68, 70: 69, 
        71: 70, 72: 71, 73: 72, 74: 73, 75: 74, 76: 75, 77: 76, 78: 77, 79: 78, 80: 79, 
        81: 80, 82: 81, 83: 82, 84: 83, 85: 84, 86: 85, 87: 86, 88: 87, 89: 88, 90: 89, 
        91: 90, 92: 91, 93: 92, 94: 93, 95: 94, 96: 95, 97: 96, 98: 97, 99: 98, 100: 99, 
        101: 100, 102: 101, 103: 102, 104: 103, 105: 104, 106: 105, 107: 106, 108: 107, 109: 108, 110: 109, 
        111: 110, 112: 111, 113: 112, 114: 113, 115: 114, 116: 115, 117: 116, 118: 117, 119: 118, 120: 119, 
        121: 120, 122: 121, 123: 122, 124: 123, 125: 124, 126: 125, 127: 126, 128: 127, 129: 128, 130: 129, 
        131: 130, 132: 131, 133: 132, 134: 133, 135: 134, 136: 135, 137: 136, 138: 137, 139: 138, 140: 139, 
        141: 140, 142: 141, 143: 142, 144: 143, 145: 144, 146: 145, 147: 146, 148: 147, 149: 148, 150: 149, 
        151: 150, 152: 151, 153: 152, 154: 153, 155: 154, 156: 155, 157: 156, 158: 157, 159: 158, 160: 159, 
        161: 160, 162: 161, 163: 162, 164: 163, 165: 164, 166: 165, 167: 166, 168: 167, 169: 168, 170: 169, 
        171: 170, 172: 171, 173: 172, 174: 173, 175: 174, 176: 175, 177: 176, 178: 177, 179: 178, 180: 179, 
        181: 180 
    }
    
    # 填充映射表
    for k, v in valid_mapping.items():
        # 注意：TrainID 通常是 0-170 (共171类)
        # 如果原始数据中有超过 171 个有效类，可能需要截断
        # 这里的 v 如果 > 170, 我们在下面的 mask 中处理，或者保持原样如果模型支持更多类
        # 针对 Text4Seg/COCO-Stuff 标准，通常只取前 171 类 (0-170)
        # 但为了防止报错，我们先填入，如果 v > 170，后续可视情况处理
        # 这里我们直接填入，确保 index 177 能够被映射
        if k < 256:
            mapping[k] = v

    # 特别处理：如果 Text4Seg 严格要求 TrainID <= 170
    # 我们可以把超过 170 的设为 255。但如果不确定，先保留映射，避免越界是最重要的。
    return mapping

def convert_folder(folder_path, output_root):
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist, skipping.")
        return

    print(f"Processing {folder_path}...")
    mapper = get_label_mapping()
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    
    # 使用 ProcessPoolExecutor 加速 (可选)
    # 为了简单稳定，继续用单线程 tqdm
    for file_name in tqdm(files):
        file_path = os.path.join(folder_path, file_name)
        save_path = os.path.join(output_root, file_name.replace(".png", "_labelTrainIds.png"))
        
        # 如果目标文件已存在，可以选择跳过以节省时间
        if os.path.exists(save_path):
            continue

        try:
            img = Image.open(file_path)
            img_np = np.array(img)
            
            # 核心修复：直接利用 numpy 的索引映射，不再会有越界问题
            # 因为 mapper 是 256 长度，而 uint8 图片最大就是 255
            img_mapped = mapper[img_np]
            
            Image.fromarray(img_mapped).save(save_path)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print(f"Error: {DATA_ROOT} not found. Please check folder name.")
    else:
        convert_folder(TRAIN_DIR, DATA_ROOT)
        convert_folder(VAL_DIR, DATA_ROOT)
        print("Conversion finished!")