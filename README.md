# 从零实现 Transformer (Encoder-Decoder)

本项目是《大模型基础与应用》课程的期中作业。我们从零开始，仅使用 PyTorch 和 NumPy，手动实现了一个完整的 **Encoder-Decoder Transformer** 架构。

为了验证模型的有效性，我们在 **IWSLT 2017 (EN-DE)** 数据集上训练了一个英语到德语的机器翻译任务，并成功复现了模型的收敛过程。

## 1. 环境设置

本项目基于 Python 3.9 和 PyTorch 开发。

1.  **克隆仓库**
    ```bash
    git clone [https://github.com/wg230605/transformer-from-scratch](https://github.com/wg230605/transformer-from-scratch)
    cd transformer-from-scratch
    ```

2.  **创建 Conda 环境**
    ```bash
    conda create -n py39 python=3.9
    conda activate py39
    ```

3.  **安装依赖**
    所有依赖项均在 `requirements.txt` 文件中。
    ```bash
    pip install -r requirements.txt
    ```

## 2. 数据准备

本项目使用 IWSLT 2017 (EN-DE) 数据集。由于 `datasets` 库的脚本问题，我们采用**手动下载**的方式。

1.  **下载数据**
    请从以下链接下载 `en-de.zip` 压缩包：
    [https://huggingface.co/datasets/iwslt2017/resolve/main/data/2017-01-trnted/texts/en/de/en-de.zip](https://huggingface.co/datasets/iwslt2017/resolve/main/data/2017-01-trnted/texts/en/de/en-de.zip)

2.  **解压数据**
    代码将自动在 `data/iwslt2017-en-de/` 路径下查找数据。请按以下结构解压文件：
    ```
    transformer-from-scratch/
    |-- data/
    |   |-- iwslt2017-en-de/
    |       |-- train.tags.de-en.en
    |       |-- train.tags.de-en.de
    |       |-- IWSLT17.TED.dev2010.de-en.en.xml
    |       |-- IWSLT17.TED.dev2010.de-en.de.xml
    |       |-- (以及其他 tst... xml 文件)
    |-- src/
    |-- configs/
    |-- ...
    ```

3.  **准备分词器**
    无需手动操作。在第一次运行时，`data_loader.py` 会自动在 `src/` 目录下训练并保存一个 `tokenizer_en_de.json` 文件。

## 3. 如何运行 (Exact Command Line)

本项目的所有超参数（包括随机种子 `seed: 42`）均已在 `configs/config.json` 文件中定义。

必须在**项目根目录** (`transformer-from-scratch/`)下运行以下命令，才能确保 Python 正确找到 `src` 模块。

### A. 复现主实验

1.  **检查模型代码**: 确保 `src/model.py` 中 `encode` 和 `decode` 方法内的 `self.pos_enc(x)` **没有**被注释。

2.  **检查训练脚本**: 打开 `src/train.py`，定位到 `main()` 函数中的“创建保存目录”部分。确保**主实验**的路径是**激活**的（未被注释），而消融实验的路径是**被注释**的：
    ```python
    # 1. 创建保存目录(正常情况）
    checkpoint_dir_path = os.path.join(root_dir, "checkpoints")
    results_dir_path = os.path.join(root_dir, "results")

    # 1. 创建保存目录(消融实验，去掉位置编码）
    # checkpoint_dir_path = os.path.join(root_dir, "checkpoints")
    # results_dir_path = os.path.join(root_dir, "results/results_no_pe")
    
    # ... 同样, 确保 "best_model_translation.pt" 路径被激活
    model_path = os.path.join(checkpoint_dir_path, "best_model_translation.pt")
    # model_path = os.path.join(checkpoint_dir_path, "best_model_translation_no_pe.pt") # 消融实验
    
    # ... 同样, 确保 "training_history_translation.json" 路径被激活
    # history_path = os.path.join(results_dir_path, "training_history_translation.json")
    history_path = os.path.join(results_dir_path, "training_history_translation_no_pe.json") # 消融实验
    ```

3.  **运行命令**:
    ```bash
    python -m src.train
    ```
    *（使用 `python -m src.train` 而不是 `python src/train.py`，这是为了让 Python 将根目录添加到搜索路径，从而正确执行 `from src.model ...` 导入。）*

4.  **查看结果**: 训练曲线图将保存在 `results/training_curves.png`。

### B. 复现消融实验 

1.  **修改模型代码**: 打开 `src/model.py`，找到 `TransformerModel` 类，注释掉 `encode` 和 `decode` 方法中的位置编码行：
    ```python
    # ... in encode() ...
    # x = self.pos_enc(x) # <-- 注释掉
    # ... in decode() ...
    # x = self.pos_enc(x) # <-- 注释掉
    ```

2.  **修改训练脚本**: 打开 `src/train.py`，定位到 `main()` 函数中。确保**消融实验**的路径是**激活**的，而主实验的路径是**被注释**的：
    ```python
    # 1. 创建保存目录(正常情况）
    # checkpoint_dir_path = os.path.join(root_dir, "checkpoints")
    # results_dir_path = os.path.join(root_dir, "results")

    # 1. 创建保存目录(消融实验，去掉位置编码）
    checkpoint_dir_path = os.path.join(root_dir, "checkpoints")
    results_dir_path = os.path.join(root_dir, "results/results_no_pe")

    # ... 同样, 确保 "best_model_translation_no_pe.pt" 路径被激活
    # model_path = os.path.join(checkpoint_dir_path, "best_model_translation.pt")
    model_path = os.path.join(checkpoint_dir_path, "best_model_translation_no_pe.pt") # 消融实验（去掉位置编码）

    # ... 同样, 确保 "training_history_translation_no_pe.json" 路径被激活
    # history_path = os.path.join(results_dir_path, "training_history_translation.json")
    history_path = os.path.join(results_dir_path, "training_history_translation_no_pe.json") # 消融实验（去掉位置编码）
    ```

3.  **运行命令**:
    ```bash
    python -m src.train
    ```

4.  **查看结果**: 训练曲线图将保存在 `results/results_no_pe/training_curves.png`。

## 4. 硬件要求

* **GPU**: 实验在 `NVIDIA A40` (40GB VRAM) 上运行。理论上，由于模型较小 (`d_model=128`) 且 `batch_size=32`，任何 VRAM 大于 8GB 的 CUDA 显卡（如 RTX 3070 / 4060）都应能成功训练。
* **训练时长**:
    * 主实验 (100 轮): 约 5.9 小时 (21309 秒)。
    * 消融实验 (53 轮, 早停): 约 3.3 小时 (11830 秒)。