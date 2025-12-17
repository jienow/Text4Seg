import ijson
import os

# --- 1. 自定义处理函数 ---
def my_process_function(item):
    """
    在这里编写你的逻辑。
    item 是流式读取到的每一个单独的字典对象。
    """
    # 示例：比如打印 item 的 ID 或者做某些计算
    # print(f"正在处理 ID: {item.get('id', 'Unknown')}")
    
    # 示例：假设我们要给每个数据加一个标记
    if 'others*256' in item.get('conversations')[-1]['value']:
        return True
    return False

# --- 2. 流式读取与处理主逻辑 ---
def process_large_json_stream(input_file, output_file=None):
    Sum = 0
    print(f"开始流式处理文件: {input_file}")
    
    # 如果需要保存结果，建议使用流式写入（一行一个JSON，即 JSONL 格式），
    # 或者像下面这样手动构建一个 JSON 数组结构写入，以保持内存低占用。
    
    if output_file:
        f_out = open(output_file, 'w', encoding='utf-8')
        f_out.write('[\n') # 手动写入 JSON 数组开头
    
    try:
        # 使用 'rb' (二进制读取) 模式打开，ijson 效率最高
        with open(input_file, 'rb') as f:
            
            # ijson.items(file, prefix)
            # 'item' 表示列表中的每一项。如果你的 JSON 是一个巨大的列表 [{}, {}, ...]，这里就填 'item'
            # 如果你的 JSON 是 {"data": [{}, {}]}，这里可能需要填 'data.item'
            parser = ijson.items(f, 'item')
            
            first = True
            count = 0
            
            for item in parser:
                # --- 调用你的处理函数 ---
                if my_process_function(item):
                    Sum += 1
                
                count += 1
                if count % 1000 == 0:
                    print(f"已处理 {count} 项...", end='\r')

            print(f"\n处理完成，共计 {count} 项。")
        print(f"包含 'others*256' 的项有 {Sum} 个。")

    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        if output_file:
            f_out.write('\n]') # 闭合 JSON 数组
            f_out.close()
            print(f"结果已保存至: {output_file}")

# --- 运行 ---
if __name__ == "__main__":
    # 你的大文件路径
    INPUT_PATH = "/mnt/data/home/lilanting/shenjie/code/Text4SegHub/playground/data/json_files/pest24_text4seg_train_combined.json" 
            
    process_large_json_stream(INPUT_PATH)