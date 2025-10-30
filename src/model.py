import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    实现缩放点积注意力计算
    """
    d_k = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    实现多头注意力机制
    """

    def __init__(self, d_model, h):
        """
        初始化多头注意力模块
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "d_model 必须能被 h 整除"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        前向传播
        """
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # if mask is not None:
        #     mask = mask.unsqueeze(1)

        x, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(x)

        return output


class PositionWiseFeedForward(nn.Module):
    """
    实现逐位置前馈网络
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化 FFN 模块
        """
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SublayerConnection(nn.Module):
    """
    实现 "Add & Norm" (残差连接与层归一化)
    """

    def __init__(self, d_model, dropout=0.1):
        """
        初始化
        """
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        前向传播
        """
        return self.norm(x + self.dropout(sublayer(x)))


class PositionalEncoding(nn.Module):
    """
    实现正弦/余弦位置编码
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        初始化位置编码模块
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * \
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播
        """
        pe_to_add = self.pe[:, :x.size(1)].requires_grad_(False)
        x = x + pe_to_add
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    实现一个编码器层 (Encoder Layer)
    """

    def __init__(self, d_model, h, d_ff, dropout=0.1):
        """
        初始化
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.d_model = d_model

    def forward(self, x, mask):
        """
        前向传播
        """
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer2(x, self.feed_forward)
        return x


def clones(module, N):
    """
    生成 N 个完全相同的层
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    实现 Encoder (N 层 EncoderLayer 堆叠)
    """

    def __init__(self, layer, N):
        """
        初始化
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, mask):
        """
        前向传播
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    实现一个解码器层 (Decoder Layer)
    """

    def __init__(self, d_model, h, d_ff, dropout=0.1):
        """
        初始化
        """
        super(DecoderLayer, self).__init__()
        # 1. 掩码多头自注意力 (Masked Multi-Head Self-Attention)
        self.self_attn = MultiHeadAttention(d_model, h)
        # 2. 多头交叉注意力 (Multi-Head Cross-Attention)
        self.cross_attn = MultiHeadAttention(d_model, h)
        # 3. 逐位置前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        # 3个 "Add & Norm"
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

        self.d_model = d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        前向传播

        参数:
        x (Tensor): 目标序列 (来自 Decoder 的上一个输出)
        memory (Tensor): 编码器的输出 (K 和 V)
        src_mask (Tensor): 源序列掩码 (用于交叉注意力)
        tgt_mask (Tensor): 目标序列掩码 (用于自注意力)
        """

        # 1. 应用掩码多头自注意力 (Q, K, V 都是 x)
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # 2. 应用多头交叉注意力 (Q 是 x, K 和 V 来自 memory)
        x = self.sublayer2(x, lambda x: self.cross_attn(x, memory, memory, src_mask))

        # 3. 应用 FFN
        x = self.sublayer3(x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    实现 Decoder (N 层 DecoderLayer 堆叠)
    """

    def __init__(self, layer, N):
        """
        初始化
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        前向传播
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class TransformerModel(nn.Module):
    """
    一个完整的 Encoder-Decoder Transformer 模型
    """

    def __init__(self, vocab_size, d_model, N, h, d_ff, dropout=0.1, max_len=5000):
        """
        初始化
        """
        super(TransformerModel, self).__init__()

        # 实例化 Encoder 和 Decoder 的"模板层"
        enc_layer = EncoderLayer(d_model, h, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, h, d_ff, dropout)

        # 1. 编码器
        self.encoder = Encoder(enc_layer, N)
        # 2. 解码器
        self.decoder = Decoder(dec_layer, N)

        # 3. 嵌入层 (源和目标)
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)

        # 4. 位置编码
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # 5. 最终的线性投影层 (Generator)
        self.generator = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

    def encode(self, src, src_mask):
        """
        编码源序列
        """
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        """
        解码目标序列
        """
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        前向传播 (用于训练)
        """
        # 1. 编码
        memory = self.encode(src, src_mask)
        # 2. 解码
        decoder_output = self.decode(tgt, memory, src_mask, tgt_mask)
        # 3. 投影到词汇表
        return self.generator(decoder_output)