import config as config
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import config

class Multi_Head_Attention(nn.Module):
    def __init__(self, config):
        # 输入输出均为 d_model  v_dim = d_model / head_num
        super(Multi_Head_Attention, self).__init__()
        self.head_num = config.head_num
        self.k_dim = config.k_dim
        self.q_linear = nn.Linear(config.d_model, config.d_model)
        self.k_linear = nn.Linear(config.d_model, config.d_model)
        self.v_linear = nn.Linear(config.d_model, config.d_model)
        self.q_multi_linear = nn.Linear(config.d_model, config.head_num * config.k_dim)
        self.k_multi_linear = nn.Linear(config.d_model, config.head_num * config.k_dim)
        self.v_multi_linear = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.p_drop)

    def forward(self, x, y, mask=None):
        # x(batch_size, seq_len, encoder_output_dim)

        query = self.q_linear(x)
        key = self.k_linear(y)
        value = self.v_linear(y)
        query = self.q_multi_linear(query)
        key = self.k_multi_linear(key)  # (batch_size, seq_len, head_num * k_dim)
        value = self.v_multi_linear(value)
        query = query.view(query.shape[0], query.shape[1], self.head_num, -1)
        key = key.view(key.shape[0], key.shape[1], self.head_num, -1)
        value = value.view(value.shape[0], value.shape[1], self.head_num, -1)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        qk_result = query.matmul(key)
        qk_result = qk_result / math.sqrt(self.k_dim)  # (batch_Size, head_num, seq_len, seq_len)
        if mask is not None:
            print(qk_result.shape, qk_result.device)
            print(mask.shape, mask.device)
            qk_result = qk_result.masked_fill(mask, 1e-10)
        qk_result = F.softmax(qk_result, dim=3)  # (batch_size,head_num,q_seq_len,k_seq_len)
        qkv_result = qk_result.matmul(value)  # (batch_size,head_num,q_seq_len,v_dim)
        qkv_result = qkv_result.permute(0, 2, 1, 3)
        qkv_result = qkv_result.view(qkv_result.shape[0], qkv_result.shape[1], -1)
        qkv_result = self.dropout(qkv_result)
        return qkv_result


class Position_Wise_Network(nn.Module):
    # FCNN encoder的第二层
    def __init__(self, config):
        super(Position_Wise_Network, self).__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.p_drop)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Position_Encoding(nn.Module):
    # 返回值：（seq_len, d_model）

    def __init__(self, config):
        super(Position_Encoding, self).__init__()
        self.max_len = config.max_len
        self.pos_linear = nn.Linear(config.max_len, config.d_model)

    def forward(self, pos):
        # identity_matrix.requires_grad =False
        result = self.pos_linear(pos)
        return result


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.atten = Multi_Head_Attention(config)
        self.att_layer_norm = nn.LayerNorm(config.d_model)
        self.position_layer = Position_Wise_Network(config)
        self.pos_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, src, mask):
        # (batch_size, seq_len, d_model)
        temp = self.atten(src, src, mask)
        src = temp + src
        src = self.att_layer_norm(src)
        temp = self.position_layer(src)
        src = temp + src
        src = self.pos_layer_norm(src)
        return src


class Encoder(nn.Module):
    def __init__(self, config, embedding_num):
        # embedding_num 源端单词的总数量
        super(Encoder, self).__init__()
        self.num_layer = config.num_layer
        self.max_len = config.max_len
        self.embedding = nn.Embedding(embedding_num, config.d_model)
        self.position_encoding = Position_Encoding(config)
        self.dropout = nn.Dropout(config.p_drop)
        self.encoder_layer = nn.ModuleList([EncoderLayer(config) for i in range(config.num_layer)])

    def forward(self, src, mask, device):
        src = self.embedding(src)  # (batch,seq_len,d_model)
        pos = torch.eye(src.shape[1], self.max_len)
        pos.requires_grad = False
        pos = pos.to(device)
        position = self.position_encoding(pos)  # (seq_len, d_model)
        src = src + position
        mask = mask.unsqueeze(1)
        mask = mask.unsqueeze(1)  # (batch_size, 1, 1,seq_len)
        #print(mask.shape)
        #print(src.shape)
        src = self.dropout(src)  # (batch,seq_len,d_model)
        for i in range(self.num_layer):
            src = self.encoder_layer[i](src, mask)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head = Multi_Head_Attention(config)
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.dec_enc_attention = Multi_Head_Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.position_wise_network = Position_Wise_Network(config)
        self.layer_norm3 = nn.LayerNorm(config.d_model)

    def forward(self, trg, encoder_output, trg_mask, src_mask):
        temp = self.masked_multi_head(trg, trg, trg_mask)
        trg = trg + temp
        trg = self.layer_norm1(trg)
        temp = self.dec_enc_attention(trg, encoder_output, src_mask)
        trg = trg + temp
        trg = self.layer_norm2(trg)
        temp = self.position_wise_network(trg)
        trg = temp + trg
        trg = self.layer_norm3(trg)
        return trg




class Decoder(nn.Module):
    def __init__(self, config, embedding_num):
        super(Decoder, self).__init__()
        self.config = config
        self.embedding_num = embedding_num
        self.embedding = nn.Embedding(embedding_num, config.d_model)
        self.position_encoding = Position_Encoding(config)  # 这里需要更改一下position_encoding的类方法
        self.dropout = nn.Dropout(config.p_drop)
        self.decoder_layer = nn.ModuleList([DecoderLayer(config) for i in range(config.num_layer)])
        self.transform_linear = nn.Linear(config.d_model, embedding_num)

    def forward(self, encoder_output, trg, trg_mask, src_mask):
        #  trg为真实输出，这里我们进行并行计算
        trg = self.embedding(trg)  # (batch,query_seq_len,d_model)
        pos = torch.eye(trg.shape[1], self.max_len)
        pos.requires_grad = False
        position = self.position_encoding(pos)  # (seq_len, d_model)
        trg = trg + position
        trg = self.dropout(trg)
        #(batch, seq_len, d_model)
        # trg_mask = torch.eye(trg.shape[1], trg.shape[1]) # 不能自己计算的！
        src_mask = src_mask.unsqueeze(1)
        for i in range(self.config.num_layer):
            trg = self.decoder_layer[i](trg, encoder_output, trg_mask, src_mask)
        output_prob = self.transform_linear(trg)
        return output_prob

class Seq2Seq(nn.Module):
    def __init__(self, config, encoder_embedding_num, decoder_embedding_num):
        super(Seq2Seq, self).__init__()
        self.config = config
        self.encoder = Encoder(config, encoder_embedding_num)
        self.decoder = Decoder(config, decoder_embedding_num)

    def compute_mask(self, to_compute, PAD_TOKEN):
        # (batch_size,seq_len)
        result = (to_compute == PAD_TOKEN)
        return result

    def forward(self, src, trg, PAD_TOKEN, device):
        # src = （seq_len,batch_size)
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)
        pos = torch.eye(trg.shape[1], self.config.max_len)
        pos.requires_grad = False
        trg_mask = torch.eye(trg.shape[1], trg.shape[1])
        src_mask = self.compute_mask(src, PAD_TOKEN)
        encoder_output = self.encoder(src, src_mask, device)
        output_prob = self.decoder(encoder_output, trg, trg_mask, src_mask)
        return output_prob





