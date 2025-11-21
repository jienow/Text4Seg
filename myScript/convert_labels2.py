import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# === 配置路径 ===
DATA_ROOT = "playground/data/coco_stuff/annotations" 
TRAIN_DIR = os.path.join(DATA_ROOT, "train2017")
VAL_DIR = os.path.join(DATA_ROOT, "val2017")

# 全局变量用于进程间共享映射表
global_mapper = None

def get_label_mapping():
    """生成修复后的映射表：将 >170 的 ID 强制设为 255"""
    mapping = np.ones(256, dtype=np.uint8) * 255
    
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
    
    for k, v in valid_mapping.items():
        if v <= 170:
            mapping[k] = v
        else:
            mapping[k] = 255 

    return mapping

def init_worker(mapper):
    """每个进程初始化时加载一次 mapper，避免重复传递"""
    global global_mapper
    global_mapper = mapper

def process_single_image(args):
    """单个图片处理函数"""
    file_path, save_path = args
    
    try:
        # 打开图片
        img = Image.open(file_path)
        img_np = np.array(img)
        
        # 使用共享的 mapper 进行极速映射
        img_mapped = global_mapper[img_np]
        
        # 保存
        Image.fromarray(img_mapped).save(save_path)
        return True
    except Exception as e:
        # 遇到错误不中断，只打印
        print(f"Error processing {file_path}: {e}")
        return False

def convert_folder_parallel(folder_path, output_root, num_workers=None):
    if not os.path.exists(folder_path):
        return

    # 如果没有指定 workers 数量，默认使用 CPU 核心数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"Processing {folder_path} with {num_workers} workers...")
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    
    # 准备任务列表
    tasks = []
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        save_path = os.path.join(output_root, file_name.replace(".png", "_labelTrainIds.png"))
        tasks.append((file_path, save_path))
    
    # 获取映射表
    mapper = get_label_mapping()

    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(mapper,)) as executor:
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), unit="img"))

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print(f"Error: {DATA_ROOT} not found. Please check folder name.")
    else:
        # 建议先处理验证集（文件少，快速验证效果）
        convert_folder_parallel(VAL_DIR, DATA_ROOT)
        
        # 再处理训练集（文件多，全速跑）
        convert_folder_parallel(TRAIN_DIR, DATA_ROOT)
        
        print("Parallel conversion finished!")