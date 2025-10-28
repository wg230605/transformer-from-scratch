import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

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


# 2. 训练分词器 (如果不存在)
def get_tokenizer(dataset, vocab_size=10000):
    """
    训练或加载一个 WordLevel 分词器
    """
    tokenizer_path = "tokenizer.json"

    if os.path.exists(tokenizer_path):
        print(f"加载已有的分词器: {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)

    print("训练新的分词器...")
    # 使用 WordLevel (基于空格和标点)
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    # 定义训练器
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # 从数据集中获取文本迭代器
    def text_iterator():
        for item in dataset['train']:
            yield item['text']

    # 训练
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # 保存
    tokenizer.save(tokenizer_path)
    print(f"分词器已保存到: {tokenizer_path}")
    return tokenizer


# 3. 创建 PyTorch Dataset
class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len):
        self.texts = [item['text'] for item in texts if item['text'].strip()]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 编码
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids

        # 截断 + 添加 SOS/EOS
        # 减 2 是为了给 [SOS] 和 [EOS] 留出空间
        token_ids = token_ids[:self.max_seq_len - 2]
        token_ids = [self.tokenizer.token_to_id(SOS_TOKEN)] + \
                    token_ids + \
                    [self.tokenizer.token_to_id(EOS_TOKEN)]

        return torch.tensor(token_ids, dtype=torch.long)


# 4. 定义 Collate Function (用于打包 Batch)
class LanguageModelCollate:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        """
        处理一个批次的数据
        batch: 一个列表, 列表中的每个元素是 __getitem__ 返回的 tensor
        """

        # 1. 获取批次中每个序列的长度
        lengths = torch.tensor([len(seq) for seq in batch], dtype=torch.long)
        max_len = lengths.max().item()

        # 2. 创建一个 (batch_size, max_len) 的全 0 张量, 用于存放 padding 后的序列
        padded_batch = torch.full(
            (len(batch), max_len),
            self.pad_token_id,
            dtype=torch.long
        )

        # 3. 将数据复制到 padded_batch 中
        for i, seq in enumerate(batch):
            padded_batch[i, :len(seq)] = seq

        # 4. 创建 src 和 tgt (用于语言建模)
        # src 是 [SOS, A, B, C, EOS, PAD, PAD]
        # tgt 是 [A, B, C, EOS, PAD, PAD, PAD]
        # 我们使用 padded_batch 作为 src
        src = padded_batch

        # tgt 是 src 向左平移一位
        # 我们用 self.pad_token_id 来填充最后一个位置
        tgt = torch.roll(src, shifts=-1, dims=1)
        tgt[:, -1] = self.pad_token_id  # 最后一个 token 的目标设为 PAD

        # 5. 创建 padding mask
        # 掩码中, PAD 的位置是 0 (False), 非 PAD 的位置是 1 (True)
        # 形状: (batch_size, 1, seq_len)
        src_mask = (src != self.pad_token_id).unsqueeze(1)  #

        return src, tgt, src_mask


# 5. 主函数: 加载数据并返回 DataLoaders
def get_dataloaders(batch_size, max_seq_len=128, vocab_size=10000):
    """
    加载 WikiText-2  数据集并创建 DataLoaders
    """
    # 1. 加载数据集
    print("从本地文件加载 WikiText-2 数据集...")
    # 定义本地数据路径
    data_path = os.path.join(root_dir, "data", "wikitext")

    # 定义每个 split 对应的文件
    data_files = {
        "train": os.path.join(data_path, "wikitext-train.arrow"),
        "validation": os.path.join(data_path, "wikitext-validation.arrow"),
        "test": os.path.join(data_path, "wikitext-test.arrow")
    }

    # 告诉 load_dataset 加载 "arrow" 格式的文件
    dataset = load_dataset("arrow", data_files=data_files)

    # 2. 获取/训练分词器
    tokenizer = get_tokenizer(dataset, vocab_size)

    pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
    vocab_size = tokenizer.get_vocab_size()  # 获取真实的词表大小

    # 3. 创建 Dataset
    train_dataset = WikiTextDataset(dataset['train'], tokenizer, max_seq_len)
    val_dataset = WikiTextDataset(dataset['validation'], tokenizer, max_seq_len)

    # 4. 创建 Collate Function 实例
    collate_fn = LanguageModelCollate(pad_token_id)

    # 5. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("数据加载完成。")
    return train_loader, val_loader, vocab_size, pad_token_id


if __name__ == '__main__':
    # 测试代码
    train_loader, val_loader, vocab_size, pad_id = get_dataloaders(batch_size=8)
    print(f"词汇表大小: {vocab_size}, PAD ID: {pad_id}")

    # 取一个批次的数据
    src, tgt, src_mask = next(iter(train_loader))

    print(f"src 形状: {src.shape}")
    print(f"tgt 形状: {tgt.shape}")
    print(f"src_mask 形状: {src_mask.shape}")

    print("\nSRC (输入):")
    print(src[0])
    print("\nTGT (目标):")
    print(tgt[0])
    print("\nMASK (掩码):")
    print(src_mask[0])