import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    实现缩放点积注意力计算

    参数:
    Q (Tensor): Query 矩阵, 形状为 (batch_size, n_heads, seq_len, d_k)
    K (Tensor): Key 矩阵, 形状为 (batch_size, n_heads, seq_len, d_k)
    V (Tensor): Value 矩阵, 形状为 (batch_size, n_heads, seq_len, d_v)

    mask (Tensor, optional): 掩码矩阵, 形状为 (batch_size, 1, seq_len, seq_len) 或 (batch_size, 1, 1, seq_len)。

    返回:
    (Tensor, Tensor): (注意力输出, 注意力权重)
    """

    # 1. & 2. 计算得分 (Q * K^T) 并进行缩放
    # Q 形状: (batch_size, n_heads, seq_len_q, d_k)
    # K.transpose(-2, -1) 形状: (batch_size, n_heads, d_k, seq_len_k)
    # scores 形状: (batch_size, n_heads, seq_len_q, seq_len_k)
    d_k = K.size(-1)  # 获取 K 的最后一个维度大小 [cite: 65]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 3. 应用掩码 (Masking)
    # 掩码的目的是在计算 softmax 之前，将特定位置的得分设置为一个非常小的负数
    # 这样在 softmax 之后，这些位置的权重将趋近于 0
    if mask is not None:
        # scores.masked_fill_ 会原地修改 scores
        # 将 mask 中为 0 (或 False) 的位置填充为 -1e9
        scores = scores.masked_fill(mask == 0, -1e9)

    # 4. 归一化 (Softmax)
    # 沿最后一个维度 (seq_len_k) 进行 softmax，得到注意力权重
    attn_weights = F.softmax(scores, dim=-1)

    # 5. 加权求和 (Weights * V)
    # attn_weights 形状: (batch_size, n_heads, seq_len_q, seq_len_k)
    # V 形状: (batch_size, n_heads, seq_len_v, d_v) (seq_len_k == seq_len_v)
    # output 形状: (batch_size, n_heads, seq_len_q, d_v)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    实现多头注意力机制
    """

    def __init__(self, d_model, h):
        """
        初始化多头注意力模块

        参数:
        d_model (int): 模型的总维度 (必须能被 h 整除)
        h (int): 注意力头的数量
        """
        super(MultiHeadAttention, self).__init__()

        # 确保 d_model 可以被 h 整除
        assert d_model % h == 0, "d_model 必须能被 h 整除"

        self.d_model = d_model  # 模型的总维度
        self.h = h  # 头的数量
        self.d_k = d_model // h  # 每个头的维度 (d_k = d_v)

        # 定义线性投影层 W_q, W_k, W_v 和 W_o
        # 我们使用一个大的线性层来一次性计算所有头的投影
        # 也可以使用 h 个小的线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        前向传播

        参数:
        query (Tensor): Query, 形状 (batch_size, seq_len_q, d_model)
        key (Tensor): Key, 形状 (batch_size, seq_len_k, d_model)
        value (Tensor): Value, 形状 (batch_size, seq_len_v, d_model)
                       (在自注意力中, seq_len_q = seq_len_k = seq_len_v)
        mask (Tensor, optional): 掩码

        返回:
        (Tensor): 多头注意力的输出, 形状 (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # 1. & 2. 线性投影 + 拆分头
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k)
        # -> (batch_size, h, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 3. 并行计算缩放点积注意力
        # Q, K, V 形状: (batch_size, h, seq_len, d_k)
        # mask 需要被正确地广播到 (batch_size, h, seq_len, seq_len)
        if mask is not None:
            # unsqueeze(1) 将 mask 变为 (batch_size, 1, ...) 以便广播到 h 个头
            mask = mask.unsqueeze(1)

            # x 形状: (batch_size, h, seq_len_q, d_k)
        # attn_weights 形状: (batch_size, h, seq_len_q, seq_len_k)
        x, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 4. 合并头
        # (batch_size, h, seq_len_q, d_k) -> (batch_size, seq_len_q, h, d_k)
        # -> (batch_size, seq_len_q, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. 最终投影
        # (batch_size, seq_len_q, d_model) -> (batch_size, seq_len_q, d_model)
        output = self.W_o(x)

        return output


class PositionWiseFeedForward(nn.Module):
    """
    实现逐位置前馈网络
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化 FFN 模块

        参数:
        d_model (int): 模型的输入和输出维度
        d_ff (int): 隐藏层的维度 (d_model * 4)
        dropout (float): Dropout 的概率
        """
        super(PositionWiseFeedForward, self).__init__()

        # 定义两层线性变换和 dropout
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        参数:
        x (Tensor): 输入张量, 形状 (batch_size, seq_len, d_model)

        返回:
        (Tensor): 输出张量, 形状 (batch_size, seq_len, d_model)
        """

        # 1. 扩展 (d_model -> d_ff)
        # 2. 激活 (ReLU)
        # 3. Dropout
        # 4. 压缩 (d_ff -> d_model)

        # x 形状: (batch_size, seq_len, d_model)
        # -> (batch_size, seq_len, d_ff)
        x = F.relu(self.fc1(x))

        # -> (batch_size, seq_len, d_ff)
        x = self.dropout(x)

        # -> (batch_size, seq_len, d_model)
        x = self.fc2(x)

        return x


class SublayerConnection(nn.Module):
    """
    实现 "Add & Norm" (残差连接与层归一化)

    这个模块封装了子层 (sublayer)，并在其后应用 dropout、残差连接和层归一化。
    """

    def __init__(self, d_model, dropout=0.1):
        """
        初始化

        参数:
        d_model (int): 模型的维度
        dropout (float): Dropout 概率
        """
        super(SublayerConnection, self).__init__()

        # 定义层归一化
        self.norm = nn.LayerNorm(d_model)

        # 定义 Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        前向传播

        参数:
        x (Tensor): 输入张量, 形状 (batch_size, seq_len, d_model)
        sublayer (nn.Module): 要封装的子层 (例如 MultiHeadAttention 或 FFN)，
                              它必须是一个可调用对象 (callable)

        返回:
        (Tensor): "Add & Norm" 之后的输出, 形状 (batch_size, seq_len, d_model)
        """

        # 1. 应用子层 (例如 MHA 或 FFN)
        # 2. 应用 Dropout
        # 3. 应用残差连接 (x + ...)
        # 4. 应用层归一化 (self.norm(...))

        # sublayer(x) 是子层的输出
        # self.dropout(sublayer(x)) 是对子层输出应用 dropout
        # x + ... 是残差连接
        # self.norm(...) 是层归一化

        return self.norm(x + self.dropout(sublayer(x)))


class PositionalEncoding(nn.Module):
    """
    实现正弦/余弦位置编码
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        初始化位置编码模块

        参数:
        d_model (int): 模型的维度
        max_len (int): 序列的最大长度, 用于预先计算位置编码
        dropout (float): Dropout 概率
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # --- 计算位置编码 ---

        # 创建一个 (max_len, d_model) 的零矩阵, 用于存放 PE
        pe = torch.zeros(max_len, d_model)

        # 创建一个 (max_len, 1) 的位置向量 [0, 1, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 创建 (d_model/2) 的分母项 (div_term)
        # 对应公式中的 10000^(2i / d_model)
        # i 的范围是 [0, 1, ..., d_model/2 - 1]
        # 2i 的范围是 [0, 2, ..., d_model - 2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
                             (-math.log(10000.0) / d_model))

        # 应用 sin 到偶数索引 (2i)
        pe[:, 0::2] = torch.sin(position * div_term)

        # 应用 cos 到奇数索引 (2i + 1)
        pe[:, 1::2] = torch.cos(position * div_term)

        # --- 注册为 buffer ---

        # 将 pe 的形状变为 (1, max_len, d_model) 以便进行批处理广播
        pe = pe.unsqueeze(0)  #

        # 注册为 buffer。
        # buffer 是模型的状态的一部分 (会随模型保存), 但它不是可训练的参数 (不会被优化器更新)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播

        参数:
        x (Tensor): 输入的词嵌入张量, 形状 (batch_size, seq_len, d_model)

        返回:
        (Tensor): 加上了位置编码的张量, 形状 (batch_size, seq_len, d_model)
        """

        # 从 buffer 中取出预先计算好的 PE
        # self.pe 形状 (1, max_len, d_model)
        # self.pe[:, :x.size(1)] 取出 (1, seq_len, d_model) 部分
        # requires_grad=False 确保 PE 不参与梯度计算
        pe_to_add = self.pe[:, :x.size(1)].requires_grad_(False)

        # 将 PE 加到 x 上 (利用了 PyTorch 的广播机制)
        x = x + pe_to_add

        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    实现一个编码器层 (Encoder Layer)

    这一层由一个多头自注意力和一个前馈网络组成,
    并且每个子层都使用了 "Add & Norm"
    """

    def __init__(self, d_model, h, d_ff, dropout=0.1):
        """
        初始化

        参数:
        d_model (int): 模型的维度
        h (int): 多头注意力的头数
        d_ff (int): FFN 隐藏层的维度
        dropout (float): Dropout 概率
        """
        super(EncoderLayer, self).__init__()

        # 1. 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, h)

        # 2. 逐位置前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # 3. 两个 "Add & Norm" (SublayerConnection)
        # 实例化两个, 一个用于自注意力, 一个用于 FFN
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

        self.d_model = d_model

    def forward(self, x, mask):
        """
        前向传播

        参数:
        x (Tensor): 输入张量, 形状 (batch_size, seq_len, d_model)
        mask (Tensor): 掩码, 用于自注意力

        返回:
        (Tensor): 编码器层的输出, 形状 (batch_size, seq_len, d_model)
        """

        # 1. 应用多头自注意力 (第一个子层) + Add & Norm
        #    注意: self.self_attn(x, x, x, mask)
        #    Q, K, V 都是 x, 这就是 "自注意力"
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))

        # 2. 应用 FFN (第二个子层) + Add & Norm
        x = self.sublayer2(x, self.feed_forward)

        return x


def clones(module, N):
    """
    生成 N 个完全相同的层

    参数:
    module (nn.Module): 需要被克隆的模块
    N (int): 克隆的数量

    返回:
    nn.ModuleList: 包含 N 个相同模块的列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    实现 Encoder (N 层 EncoderLayer 堆叠)
    """

    def __init__(self, layer, N):
        """
        初始化

        参数:
        layer (EncoderLayer): 一个 EncoderLayer 实例, 将被克隆 N 次
        N (int): 层的数量
        """
        super(Encoder, self).__init__()

        # 使用 clones 函数克隆 N 个 layer
        self.layers = clones(layer, N)

        # 再添加一个最终的 LayerNorm
        # 这是原始论文的设计, 在 N 层之后再加一个归一化
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, mask):
        """
        前向传播

        参数:
        x (Tensor): 输入 (通常是 词嵌入 + 位置编码), 形状 (batch_size, seq_len, d_model)
        mask (Tensor): 掩码

        返回:
        (Tensor): Encoder 的最终输出, 形状 (batch_size, seq_len, d_model)
        """

        # 循环 N 次, 依次通过 N 个 EncoderLayer
        for layer in self.layers:
            x = layer(x, mask)

        # 通过最终的 LayerNorm
        return self.norm(x)


class EncoderModel(nn.Module):
    """
    一个完整的 Transformer Encoder 模型, 用于语言建模
    """

    def __init__(self, vocab_size, d_model, N, h, d_ff, dropout=0.1, max_len=5000):
        """
        初始化完整的 Encoder 模型

        参数:
        vocab_size (int): 词汇表大小
        d_model (int): 模型的维度
        N (int): EncoderLayer 的层数
        h (int): 多头注意力的头数
        d_ff (int): FFN 隐藏层的维度
        dropout (float): Dropout 概率
        max_len (int): 位置编码的最大序列长度
        """
        super(EncoderModel, self).__init__()

        self.d_model = d_model

        # 1. 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. 位置编码
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # 3. N 层 Encoder
        enc_layer = EncoderLayer(d_model, h, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)

        # 4. 输出层 (Generator)
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask):
        """
        前向传播

        参数:
        src (Tensor): 输入的词元索引, 形状 (batch_size, seq_len)
        src_mask (Tensor): 源序列的掩码

        返回:
        (Tensor): 模型的输出 (logits), 形状 (batch_size, seq_len, vocab_size)
        """

        # 1. & 2. 应用嵌入和位置编码
        # src 形状: (batch_size, seq_len)
        # x 形状: (batch_size, seq_len, d_model)
        x = self.embedding(src) * math.sqrt(self.d_model)  # 缩放
        x = self.pos_enc(x)

        # 3. 通过 N 层 Encoder
        # x 形状: (batch_size, seq_len, d_model)
        x = self.encoder(x, src_mask)

        # 4. 通过输出层
        # output 形状: (batch_size, seq_len, vocab_size)
        output = self.generator(x)

        return output