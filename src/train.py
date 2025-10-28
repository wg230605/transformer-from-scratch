import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import os
import time
import json
import matplotlib.pyplot as plt

# 从 src 目录导入我们的模块
from src.model import EncoderModel
from src.data_loader import get_dataloaders, PAD_TOKEN

# ---------------------------------
# 1. 从 JSON 文件加载配置
# ---------------------------------
# 获取当前脚本 (train.py) 的绝对路径
script_path = os.path.abspath(__file__)

# 获取 train.py 所在的目录 (即 src/ 目录)
script_dir = os.path.dirname(script_path)

# 获取项目根目录 (即 src/ 目录的上一级)
root_dir = os.path.dirname(script_dir)

# 构造 config.json 的绝对路径
CONFIG_FILE = os.path.join(root_dir, "configs", "config.json")

try:
    with open(CONFIG_FILE, 'r') as f:
        CONFIG = json.load(f)
    print("配置已加载: ", CONFIG_FILE)
except FileNotFoundError:
    print(f"错误: 配置文件 {CONFIG_FILE} 未找到。")
    exit()


# ---------------------------------
# 2. 绘图函数
# ---------------------------------
def plot_history(history, save_dir):
    """
    绘制训练和验证曲线并保存
    """
    print("正在生成训练曲线图...")

    # 确保 save_dir 存在
    os.makedirs(save_dir, exist_ok=True)

    # 1. 创建图表 (Loss vs. Val Loss)
    plt.figure(figsize=(12, 5))

    # 第一个子图: 损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 第二个子图: 困惑度
    plt.subplot(1, 2, 2)
    plt.plot(history['perplexity'], label='Validation Perplexity')
    plt.title('Validation Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)

    # 调整布局并保存
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path)
    print(f"训练曲线图已保存到: {save_path}")
    plt.close()


# ---------------------------------
# 3. 训练和评估函数 (与之前相同)
# ---------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    """
    执行一个训练 epoch
    """
    model.train()  # 切换到训练模式
    total_loss = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for src, tgt, src_mask in progress_bar:
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)

        optimizer.zero_grad()
        outputs = model(src, src_mask)

        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            tgt.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    执行评估
    """
    model.eval()  # 切换到评估模式
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating", leave=False)
        for src, tgt, src_mask in progress_bar:
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)

            outputs = model(src, src_mask)

            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                tgt.view(-1)
            )

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    try:
        perplexity = math.exp(avg_loss)  # 困惑度 PPL
    except OverflowError:
        perplexity = float('inf')  # 防止 loss 过大导致 exp 溢出

    return avg_loss, perplexity


# ---------------------------------
# 4. 主训练循环
# ---------------------------------

def main():
    device = CONFIG["device"]
    if not torch.cuda.is_available():
        print(f"警告: CUDA 不可用。强制使用 CPU。")
        device = "cpu"
    elif device.startswith("cuda"):
        try:
            # 尝试访问指定的 GPU
            torch.cuda.get_device_name(device)
            print(f"成功找到并使用设备: {device}")
        except (RuntimeError, AssertionError):
            print(f"警告: 无法找到设备 {device}。回退到 cuda:0。")
            device = "cuda:0"
            try:
                # 尝试访问 cuda:0
                torch.cuda.get_device_name(device)
            except (RuntimeError, AssertionError):
                print(f"错误: 无法找到任何 CUDA 设备。强制使用 CPU。")
                device = "cpu"
    else:
        print(f"警告: 配置的设备不是 'cuda:X'。强制使用 CPU。")
        device = "cpu"

    CONFIG["device"] = device  # 将最终确定的设备存回 CONFIG

    torch.manual_seed(CONFIG["seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed(CONFIG["seed"])

    print(f"最终使用设备: {device}")

    # 构造指向项目根目录的绝对路径
    checkpoint_dir_path = os.path.join(root_dir, "checkpoints")
    results_dir_path = os.path.join(root_dir, "results")

    print(f"检查点将保存到: {checkpoint_dir_path}")
    print(f"结果将保存到: {results_dir_path}")

    # 1. 创建目录
    os.makedirs(checkpoint_dir_path, exist_ok=True)  # <-- 新代码
    os.makedirs(results_dir_path, exist_ok=True)  # <-- 新代码

    # 2. 加载数据
    train_loader, val_loader, vocab_size, pad_id = get_dataloaders(
        batch_size=CONFIG["batch_size"],
        max_seq_len=CONFIG["max_seq_len"],
        vocab_size=CONFIG["vocab_size"]
    )
    CONFIG["vocab_size"] = vocab_size
    print(f"真实词汇表大小: {vocab_size}, PAD ID: {pad_id}")

    # 3. 实例化模型
    model = EncoderModel(
        vocab_size=vocab_size,
        d_model=CONFIG["d_model"],
        N=CONFIG["N"],
        h=CONFIG["h"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
        max_len=CONFIG["max_seq_len"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=1, factor=0.5, verbose=True
    )

    # 5. 训练循环
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "perplexity": []}
    patience_counter = 0

    print(f"开始训练... (总共 {CONFIG['epochs']} 轮, 早停忍耐 {CONFIG['patience']} 轮)")
    start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        epoch_start_time = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, perplexity = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start_time

        print(f"--- Epoch {epoch}/{CONFIG['epochs']} ---")
        print(f"Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["perplexity"].append(perplexity)

        scheduler.step(val_loss)

        # 6. 保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(checkpoint_dir_path, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"验证损失改善, 模型已保存到: {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"验证损失未改善。早停计数: {patience_counter}/{CONFIG['patience']}")

        if patience_counter >= CONFIG['patience']:
            print(f"\n--- 早停触发 ---")
            print(f"验证损失在 {CONFIG['patience']} 轮内没有改善。")
            break

    total_time = time.time() - start_time
    print(f"\n训练完成. 总耗时: {total_time:.2f}s")

    # 7. 保存训练历史和配置
    # history_path = os.path.join(CONFIG["results_dir"], "training_history.json") # <-- 旧代码
    history_path = os.path.join(results_dir_path, "training_history.json")  # <-- 新代码
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存到: {history_path}")

    # config_path = os.path.join(CONFIG["results_dir"], "config.json") # <-- 旧代码
    config_path = os.path.join(results_dir_path, "config.json")  # <-- 新代码
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"配置已保存到: {config_path}")

    # 8. 调用绘图函数
    plot_history(history, results_dir_path)


if __name__ == "__main__":
    main()