import torch
import time

def check_gpu():
    print("-" * 30)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDNN 版本: {torch.backends.cudnn.version()}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"检测到 {device_count} 张显卡:")
        print("-" * 30)
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"【显卡 {i}】: {device_name}")
            
            # 获取显存信息
            props = torch.cuda.get_device_properties(i)
            print(f"  - 总显存: {props.total_memory / 1024**3:.2f} GB")
            
            # 进行简单的张量计算测试
            try:
                print("  - 正在进行张量计算测试...", end="")
                # 在指定显卡上创建随机张量并做矩阵乘法
                device = torch.device(f'cuda:{i}')
                x = torch.randn(5000, 5000, device=device)
                y = torch.randn(5000, 5000, device=device)
                
                start = time.time()
                z = torch.matmul(x, y)
                torch.cuda.synchronize() # 等待计算完成
                end = time.time()
                
                print(f" 通过! (耗时: {end - start:.4f}秒)")
            except Exception as e:
                print(f" 失败! 错误信息: {e}")
            print("-" * 30)
    else:
        print("未检测到 GPU，请检查驱动或 CUDA 版本是否与 PyTorch 版本匹配。")

if __name__ == "__main__":
    check_gpu()