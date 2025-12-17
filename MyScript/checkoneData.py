import ijson
import json
import itertools

filename = '/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/json_files/pest24_text4seg_train_refer.json'

# 你想查看第几个数据？(注意：Python 索引从 0 开始，0 代表第 1 个，99 代表第 100 个)
target_index = 12000  # <--- 修改这里，例如改成 0, 10, 99 等

try:
    with open(filename, 'rb') as f:
        # 创建生成器
        parser = ijson.items(f, 'item')
        
        # 读取前 100 个数据放入内存中的列表
        batch_data = list(itertools.islice(parser, target_index+1))

    print(f"读取完成，内存中现有 {len(batch_data)} 条数据。")

    # 安全检查：确保你想看的下标没有超出实际读取的数据量
    if 0 <= target_index < len(batch_data):
        print(f"-" * 30)
        print(f"正在展示列表中的第 [{target_index}] 条数据：")
        # 使用数组下标访问
        selected_item = batch_data[target_index]
        print(json.dumps(selected_item, indent=4, ensure_ascii=False))
        print(f"-" * 30)
    else:
        print(f"错误：下标 {target_index} 超出了范围。当前只有 0 到 {len(batch_data)-1} 的数据。")

except FileNotFoundError:
    print(f"错误：找不到文件 {filename}")
except Exception as e:
    print(f"发生错误：{e}")