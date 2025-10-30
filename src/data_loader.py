import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import re  # 导入正则表达式库, 用于解析 XML

# --- 这部分保留, 用于定位项目根目录 ---
# 获取当前脚本 (data_loader.py) 的绝对路径
script_path = os.path.abspath(__file__)
# 获取 data_loader.py 所在的目录 (即 src/ 目录)
script_dir = os.path.dirname(script_path)
# 获取项目根目录 (即 src/ 目录的上一级)
root_dir = os.path.dirname(script_dir)

# 1. 定义特殊 Token
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
SOS_TOKEN = "[SOS]"  # Start of Sentence
EOS_TOKEN = "[EOS]"  # End of Sentence
special_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


# --- 辅助函数 (新): 用于解析文件 ---

def _parse_train_tags(file_path):
    """
    读取 train.tags.* 文件, 跳过 <tag> 行。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if not line.strip().startswith("<")]
    return lines


def _parse_xml_seg(file_path):
    """
    读取 .xml 文件, 提取 <seg ...> ... </seg> 之间的内容。
    """
    # 编译一个正则表达式来匹配 <seg ...> ... </seg>
    seg_pattern = re.compile(r"<seg.*?>(.*?)</seg>")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 找到所有匹配的句子
        lines = [match.strip() for match in seg_pattern.findall(content)]
    return lines


# --- 分词器 ---
def get_tokenizer(training_files, vocab_size=20000):
    """
    训练或加载一个共享的 WordLevel 分词器

    参数:
    training_files (list): 用于训练分词器的文件路径列表
    vocab_size (int): 词汇表大小
    """
    # 保存到src目录下
    tokenizer_path = os.path.join(script_dir, "tokenizer_en_de.json")

    if os.path.exists(tokenizer_path):
        print(f"加载已有的分词器: {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)

    print("训练新的(EN-DE)分词器...")
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    # 定义训练器
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # 从文件列表训练
    tokenizer.train(files=training_files, trainer=trainer)

    # 保存
    tokenizer.save(tokenizer_path)
    print(f"分词器已保存到: {tokenizer_path}")
    return tokenizer


# --- PyTorch Dataset ---
class TranslationDataset(Dataset):
    """
    用于机器翻译的平行语料数据集
    """

    def __init__(self, src_file, tgt_file, is_xml=False):
        """
        参数:
        src_file (str): 源语言文件路径
        tgt_file (str): 目标语言文件路径
        is_xml (bool): 文件是否为 .xml 格式 (用于验证/测试集)
        """
        print(f"加载数据 (is_xml={is_xml})...")
        if is_xml:
            self.src_lines = _parse_xml_seg(src_file)
            self.tgt_lines = _parse_xml_seg(tgt_file)
        else:
            self.src_lines = _parse_train_tags(src_file)
            self.tgt_lines = _parse_train_tags(tgt_file)

        assert len(self.src_lines) == len(self.tgt_lines), \
            f"源文件和目标文件行数不匹配! {len(self.src_lines)} != {len(self.tgt_lines)}"
        print(f"成功加载 {len(self.src_lines)} 行数据。")

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        # 返回纯文本对, 分词和张量化将在 collate_fn 中完成
        return self.src_lines[idx], self.tgt_lines[idx]

# --- 主函数 ---
def get_dataloaders(batch_size, max_seq_len=128, vocab_size=20000, lang_src="en", lang_tgt="de"):
    """
    加载 IWSLT2017 EN-DE 数据集并创建 DataLoaders
    """
    # 1. 定义数据路径
    data_path = os.path.join(root_dir, "data", "iwslt2017-en-de")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据目录: {data_path}。请确保你的 'iwslt2017-en-de' 文件夹在 'data' 目录下。")

    # 定义训练集文件
    train_src_file = os.path.join(data_path, f"train.tags.{lang_tgt}-{lang_src}.{lang_src}")  # train.tags.de-en.en
    train_tgt_file = os.path.join(data_path, f"train.tags.{lang_tgt}-{lang_src}.{lang_tgt}")  # train.tags.de-en.de

    # 定义验证集文件
    val_src_file = os.path.join(data_path,
                                f"IWSLT17.TED.dev2010.{lang_tgt}-{lang_src}.{lang_src}.xml")  # dev2010.de-en.en.xml
    val_tgt_file = os.path.join(data_path,
                                f"IWSLT17.TED.dev2010.{lang_tgt}-{lang_src}.{lang_tgt}.xml")  # dev2010.de-en.de.xml

    # 2. 获取/训练分词器 (在两个训练文件上)
    tokenizer = get_tokenizer(
        training_files=[train_src_file, train_tgt_file],
        vocab_size=vocab_size
    )
    pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
    real_vocab_size = tokenizer.get_vocab_size()

    # 3. 创建 Dataset
    train_dataset = TranslationDataset(train_src_file, train_tgt_file, is_xml=False)
    val_dataset = TranslationDataset(val_src_file, val_tgt_file, is_xml=True)

    # 4. 定义 Collate Function (用于打包 Batch, 这是 Seq2Seq 的核心)
    def translation_collate_fn(batch):
        """
        处理一个批次的 (src_text, tgt_text) 对
        """
        src_batch, tgt_batch = [], []

        # --- 1. 编码和截断 ---
        for src_text, tgt_text in batch:
            # 编码源序列
            src_ids = [tokenizer.token_to_id(SOS_TOKEN)] + \
                      tokenizer.encode(src_text).ids + \
                      [tokenizer.token_to_id(EOS_TOKEN)]
            src_batch.append(torch.tensor(src_ids[:max_seq_len], dtype=torch.long))

            # 编码目标序列
            tgt_ids = [tokenizer.token_to_id(SOS_TOKEN)] + \
                      tokenizer.encode(tgt_text).ids + \
                      [tokenizer.token_to_id(EOS_TOKEN)]
            tgt_batch.append(torch.tensor(tgt_ids[:max_seq_len], dtype=torch.long))

        # --- 2. 填充 (Padding) ---
        src_padded = torch.nn.utils.rnn.pad_sequence(
            src_batch, batch_first=True, padding_value=pad_token_id
        )  # 形状: (batch_size, src_len)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_batch, batch_first=True, padding_value=pad_token_id
        )  # 形状: (batch_size, tgt_len)

        # --- 3. 创建掩码 ---

        # (a) Encoder 的 Padding Mask (src_mask)
        # 形状: (batch_size, 1, 1, src_len)
        src_mask = (src_padded != pad_token_id).unsqueeze(1).unsqueeze(2)

        # (b) Decoder 的 Padding Mask
        # 形状: (batch_size, 1, 1, tgt_len)
        tgt_pad_mask = (tgt_padded != pad_token_id).unsqueeze(1).unsqueeze(2)

        # (c) Decoder 的 "未来" 掩码 (Look-Ahead Mask)
        tgt_len = tgt_padded.size(1)
        # torch.triu(..., diagonal=1) 创建一个上三角矩阵 (不包括对角线)
        look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()

        # (d) 合并 Decoder 的两个掩码
        # 形状: (batch_size, 1, tgt_len, tgt_len)
        tgt_mask = tgt_pad_mask & (~look_ahead_mask)  # ~ 是 "非" 运算, 反转掩码

        return src_padded, tgt_padded, src_mask, tgt_mask

    # 5. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=translation_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=translation_collate_fn
    )

    print("数据加载完成。")
    return train_loader, val_loader, real_vocab_size, pad_token_id


if __name__ == '__main__':
    # 测试代码
    print(f"项目根目录: {root_dir}")
    train_loader, val_loader, vocab_size, pad_id = get_dataloaders(
        batch_size=4,
        max_seq_len=50
    )

    print(f"词汇表大小: {vocab_size}, PAD ID: {pad_id}")

    # 取一个批次的数据
    src, tgt, src_mask, tgt_mask = next(iter(train_loader))

    print(f"\n--- 批次数据形状 ---")
    print(f"src (Encoder 输入) 形状: {src.shape}")
    print(f"tgt (Decoder 输入/目标) 形状: {tgt.shape}")
    print(f"src_mask (Encoder 掩码) 形状: {src_mask.shape}")
    print(f"tgt_mask (Decoder 掩码) 形状: {tgt_mask.shape}")

    print("\n--- Decoder 掩码示例 (批次中的第 0 个) ---")
    print(f"这是一个 {tgt_mask.shape[2]}x{tgt_mask.shape[3]} 的矩阵:")
    print(tgt_mask[0, 0, :, :])