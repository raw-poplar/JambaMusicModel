import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, TypeAlias
from functools import partial
from torch.nn.utils.rnn import pad_sequence
import logging
import os
from tqdm import tqdm

from mamba2minimal.mamba2 import Mamba2, Mamba2Config, RMSNorm, InferenceCache as MambaInferenceCache

Device: TypeAlias = str | torch.device | None

# --- Jamba S-Input S-Output Specific Configuration ---
@dataclass
class JambaSConfig(Mamba2Config):
    """
    Configuration class for JambaForAudioGeneration_S_In_S_Out model.
    Inherits Mamba2 parameters and adds Attention/audio/s-input specific parameters.
    JambaForAudioGeneration_S_In_S_Out 模型的配置类。
    继承 Mamba2 的参数，并添加了 Attention/音频/S输入相关的特定参数。
    """
    num_attention_heads: int = 8 # 注意力头的数量
    attention_dropout: float = 0.1 # 注意力层的 dropout 概率
    use_attention_layer_indices: List[int] = field(default_factory=list) # 指定哪些层使用注意力机制的索引列表
    
    dac_codebook_size: int = 1025       # Size of each DAC codebook, *including* a potential EOS token.
                                        # 每个 DAC 码本的大小，*包括*一个可能的 EOS (序列结束) 标记。
    dac_num_quantizers: int = 9         # Number of DAC quantizers (Q).
                                        # DAC 量化器的数量 (Q)。
    
    s_embedding_dim_individual: int = 64 # Embedding dimension for s_indices from each quantizer for the main model.
                                           # 主模型中每个量化器的 s_indices 的嵌入维度。
    
    # --- Style Encoder Specific Config ---
    style_dim: int = 64 # Dimension of the output style vector.
                         # 输出风格向量的维度。
    style_s_embedding_dim_individual: int = 64 # Embedding dim for s_indices in StyleEncoder.
                                               # StyleEncoder 中 s_indices 的嵌入维度。
    style_encoder_num_conv_layers: int = 4 # 风格编码器中卷积层的数量
    style_encoder_kernel_size: int = 3 # 风格编码器中卷积核的大小
    # input_channels for StyleEncoder will be dac_num_quantizers * style_s_embedding_dim_individual
    # StyleEncoder 的输入通道数将是 dac_num_quantizers * style_s_embedding_dim_individual
    attention_window_size: Optional[int] = None # 注意力滑动窗口大小，None或0表示不使用

    def __post_init__(self):
        """
        后初始化方法，用于在 dataclass 初始化后进行一些额外的设置或检查。
        """
        if hasattr(super(), '__post_init__'): # 检查父类是否有 __post_init__ 方法
            super().__post_init__() # 调用父类的 __post_init__ 方法

        # 如果 use_attention_layer_indices 未指定或为可调用对象，则根据 n_layer 自动设置
        if not self.use_attention_layer_indices or callable(self.use_attention_layer_indices):
            temp_n_layer = getattr(self, 'n_layer', 12) # 获取 n_layer 属性，默认为 12
            if temp_n_layer is None: temp_n_layer = 12
            self.use_attention_layer_indices = [i for i in range(temp_n_layer) if i % 2 != 0] # 默认使用奇数层作为注意力层

        # 检查 s_embedding_dim_individual 是否正确设置
        if not hasattr(self, 's_embedding_dim_individual') or self.s_embedding_dim_individual <= 0:
            print("Warning: s_embedding_dim_individual not properly set or invalid in JambaSConfig.")
            print("警告: JambaSConfig 中的 s_embedding_dim_individual 未正确设置或无效。")
            self.s_embedding_dim_individual = 128 # 设置默认值
        
        # 检查 style_s_embedding_dim_individual 是否正确设置
        if not hasattr(self, 'style_s_embedding_dim_individual') or self.style_s_embedding_dim_individual <= 0:
            print("Warning: style_s_embedding_dim_individual not properly set or invalid.")
            print("警告: style_s_embedding_dim_individual 未正确设置或无效。")
            self.style_s_embedding_dim_individual = 64 # 设置默认值
        
        # 检查 style_dim 是否正确设置
        if not hasattr(self, 'style_dim') or self.style_dim <= 0:
            print("Warning: style_dim not properly set.")
            print("警告: style_dim 未正确设置。")
            self.style_dim = 128 # 设置默认值


        d_model = getattr(self, 'd_model', 0) # 获取 d_model 属性，默认为 0
        # 检查 d_model 是否能被 num_attention_heads 整除
        if self.num_attention_heads > 0 and d_model % self.num_attention_heads != 0:
            print(f"Warning: d_model ({d_model}) is not divisible by num_attention_heads ({self.num_attention_heads}).")
            print(f"警告: d_model ({d_model}) 不能被 num_attention_heads ({self.num_attention_heads}) 整除。")
        elif self.num_attention_heads == 0: # 如果注意力头数量为0，但指定了注意力层索引，则发出警告
            if self.use_attention_layer_indices and any(idx >=0 for idx in self.use_attention_layer_indices):
                 print("Warning: use_attention_layer_indices contains attention layer indices, but num_attention_heads is 0.")
                 print("警告: use_attention_layer_indices 包含注意力层索引，但 num_attention_heads 为 0。")


# --- Style Encoder using S input ---
class StyleEncoderUsingS(nn.Module):
    """
    使用 S 输入（离散音频码元索引）的风格编码器。
    它将原始的 S 码元序列编码为一个固定维度的风格向量。
    """
    def __init__(self, config: JambaSConfig, device: Device = None):
        super().__init__()
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu") # 设置设备

        # Embeddings for s_indices within the style encoder
        # 风格编码器内部 s_indices 的嵌入层
        self.s_embeddings_style = nn.ModuleList([
            nn.Embedding(config.dac_codebook_size, config.style_s_embedding_dim_individual, device=self.device)
            for _ in range(config.dac_num_quantizers)
        ])
        
        # Effective input channels to the Conv1D after embedding and concatenation
        # 嵌入和拼接后，Conv1D 的有效输入通道数
        style_encoder_conv_input_channels = config.dac_num_quantizers * config.style_s_embedding_dim_individual
        
        layers = [] # 卷积层列表
        current_channels = style_encoder_conv_input_channels # 当前通道数
        # Define intermediate channels for the CNN, can be based on style_dim or input_channels
        # 定义 CNN 的中间通道数，可以基于 style_dim 或输入通道数
        intermediate_channels = max(config.style_dim * 2, style_encoder_conv_input_channels // 2) 
        if intermediate_channels == 0 : intermediate_channels = config.style_dim # 确保中间通道数不为零

        # 第一个卷积层
        layers.append(nn.Conv1d(current_channels, intermediate_channels, 
                                kernel_size=config.style_encoder_kernel_size, 
                                stride=1, padding=config.style_encoder_kernel_size//2, device=self.device))
        layers.append(nn.ReLU()) # ReLU 激活函数
        current_channels = intermediate_channels # 更新当前通道数

        # 构建后续的卷积层
        for i in range(config.style_encoder_num_conv_layers - 1):
            next_channels = min(current_channels * 2, config.style_dim * 4) # 计算下一层的通道数
            if next_channels == 0 : next_channels = config.style_dim # 确保下一层通道数不为零
            stride = 2 # 使用固定的步长 2
            layers.append(nn.Conv1d(current_channels, next_channels, 
                                    kernel_size=config.style_encoder_kernel_size, stride=stride, 
                                    padding=config.style_encoder_kernel_size//2, device=self.device))
            layers.append(nn.ReLU())
            current_channels = next_channels # 更新当前通道数
            if current_channels == 0 and i < config.style_encoder_num_conv_layers -2: # 防止通道数过早变为0
                current_channels = config.style_dim


        self.conv_net = nn.Sequential(*layers) # 将所有卷积层和激活函数组合成一个序列模块
        # The output size of conv_net depends on strides and num_layers.
        # AdaptiveAvgPool1d handles variable length after convs.
        # The input to FC layer will be `current_channels` (the last conv layer's output channels).
        # conv_net 的输出大小取决于步长和层数。
        # AdaptiveAvgPool1d 处理卷积后可变长度的输出。
        # 全连接层的输入将是 `current_channels` (最后一个卷积层的输出通道数)。
        self.fc = nn.Linear(current_channels, config.style_dim, device=self.device) # 全连接层，输出风格向量
        
        print(f"Initialized StyleEncoderUsingS on device {self.device}")
        print(f"在设备 {self.device} 上初始化 StyleEncoderUsingS")
        print(f"  StyleEncoder input s-emb dim (individual): {config.style_s_embedding_dim_individual}")
        print(f"  风格编码器输入 s-emb 维度 (单个): {config.style_s_embedding_dim_individual}")
        print(f"  StyleEncoder CNN input channels: {style_encoder_conv_input_channels}")
        print(f"  风格编码器 CNN 输入通道数: {style_encoder_conv_input_channels}")
        print(f"  StyleEncoder output style_dim: {config.style_dim}")
        print(f"  风格编码器输出 style_dim: {config.style_dim}")


    def forward(self, s_originals_list: List[Tensor], lengths: Tensor) -> Tensor:
        """
        前向传播函数。
        Args:
            s_originals_list: List of *unpadded* original s_idx tensors, each shape (T_orig, Q).
                              原始的、*未填充*的 s_idx 张量列表，每个张量的形状为 (T_orig, Q)。
            lengths: Tensor of original lengths for each sequence in s_originals_list (B,).
                     s_originals_list 中每个序列的原始长度张量，形状为 (B,)。
        Returns:
            style_vector: (B, style_dim).
                          风格向量，形状为 (B, style_dim)。
        """
        processed_s_for_conv_list = [] # 用于存储处理后送入卷积层的数据列表
        for s_item_original in s_originals_list: # s_item_original 的形状是 (T_orig, Q)
            s_item_original = s_item_original.to(self.device) # 将数据移至指定设备
            embeddings_list_item = [] # 用于存储单个样本的嵌入结果
            for q in range(self.config.dac_num_quantizers): # 遍历每个量化器
                # s_item_original[:, q] 的形状是 (T_orig)
                embeddings_list_item.append(self.s_embeddings_style[q](s_item_original[:, q])) # (T_orig, style_s_emb_dim)
            
            # Concatenate embeddings: (T_orig, Q * style_s_emb_dim)
            # 拼接嵌入结果: (T_orig, Q * style_s_emb_dim)
            concatenated_embeddings = torch.cat(embeddings_list_item, dim=-1) 
            
            # Transpose for Conv1d: (Q * style_s_emb_dim, T_orig)
            # 为 Conv1d 转置: (Q * style_s_emb_dim, T_orig)
            processed_s_for_conv_list.append(concatenated_embeddings.transpose(0, 1))

        # Pad the list of (C_eff, T_orig) tensors along the time dimension (dim=2 for pad_sequence batch_first=False)
        # pad_sequence expects (T, *) or (T, B, C) if not batch_first
        # Our list items are (C_eff, T_orig). We need to make them (T_orig, C_eff) for pad_sequence
        # 沿着时间维度 (对于 pad_sequence batch_first=False，dim=2) 填充 (C_eff, T_orig) 张量列表
        # pad_sequence 期望输入形状为 (T, *) 或 (T, B, C) (如果 batch_first=False)
        # 我们的列表项是 (C_eff, T_orig)。我们需要将它们转换为 (T_orig, C_eff) 以便 pad_sequence 处理
        s_transposed_for_padding = [s.transpose(0,1) for s in processed_s_for_conv_list] # (T_orig, C_eff) 张量列表
        
        # pad_sequence will output (T_max, B, C_eff) if batch_first=False (default)
        # 如果 batch_first=False (默认)，pad_sequence 将输出 (T_max, B, C_eff)
        s_padded_for_conv = pad_sequence(s_transposed_for_padding, batch_first=False, padding_value=0.0)
        
        # Permute to (B, C_eff, T_max) for Conv1d
        # 重排为 (B, C_eff, T_max) 以适应 Conv1d
        s_padded_for_conv = s_padded_for_conv.permute(1, 2, 0)
        
        s_padded_for_conv = s_padded_for_conv.to(self.conv_net[0].weight.device) # Ensure device
                                                                                # 确保数据在正确的设备上
        
        x = self.conv_net(s_padded_for_conv) # (B, C_last_conv, T_reduced_padded)
        
        #自适应平均池化，将可变长度的输出转换为固定大小
        pooled_x = F.adaptive_avg_pool1d(x, 1).squeeze(-1) # (B, C_last_conv)
        
        style_vector = self.fc(pooled_x) # (B, style_dim)，通过全连接层得到最终的风格向量
        return style_vector


# --- Attention Layer (copied from original, no changes needed here for s->s if d_model is consistent) ---
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现。
    """
    def __init__(self, config: JambaSConfig, device=None): # Uses JambaSConfig, 使用 JambaSConfig
        super().__init__()
        self.config = config
        self.d_model = config.d_model # 模型维度
        self.num_heads = config.num_attention_heads # 注意力头数量
        if self.num_heads == 0: 
             self.head_dim = 0 # 如果头数量为0，则头维度也为0
        else:
             assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads" # d_model 必须能被 num_heads 整除
             self.head_dim = self.d_model // self.num_heads # 计算每个头的维度

        if self.head_dim == 0: # 如果头维度为0，则所有投影层都是恒等映射
            self.q_proj = nn.Identity()
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()
            self.out_proj = nn.Identity()
        else: # 否则，定义 Q, K, V 和输出的线性投影层
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False, device=device)
            self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False, device=device)
            self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False, device=device)
            self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False, device=device)
        self.dropout = nn.Dropout(config.attention_dropout) # Dropout 层

    def forward(self, x: Tensor, past_kv: Optional[Tuple[Tensor, Tensor]] = None, causal: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        前向传播函数。
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)。
            past_kv: 可选的先前时间步的 K 和 V 张量元组，用于缓存。
            causal: 是否使用因果注意力 (causal attention)。
            attn_mask: 可选的注意力掩码。
        Returns:
            输出张量和当前的 K, V 缓存。
        """
        if self.head_dim == 0: # 如果头维度为0，直接返回输入
             return x, None
        batch_size, seq_len, _ = x.shape # 获取批次大小和序列长度
        q = self.q_proj(x) # 查询向量
        k = self.k_proj(x) # 键向量
        v = self.v_proj(x) # 值向量
        # 将 Q, K, V 重塑并转置以适应多头注意力计算
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, seq_len, head_dim)
        
        if past_kv is not None: # 如果存在过去的 K, V 缓存
            past_k, past_v = past_kv
            if isinstance(past_k, Tensor) and isinstance(past_v, Tensor): # 检查缓存类型是否正确
                window_size = getattr(self.config, 'attention_window_size', None)
                
                # k_current_step 和 v_current_step 是当前时间步的K,V (在 step 模式下 seq_len=1)
                k_current_step = k 
                v_current_step = v

                if window_size is not None and window_size > 0:
                    # past_k, past_v 的形状: (B, num_heads, past_seq_len, head_dim)
                    # k_current_step, v_current_step 的形状: (B, num_heads, current_seq_len, head_dim)
                    
                    # 如果过去的缓存已经等于或超过窗口大小，我们需要裁剪它
                    if past_k.shape[2] >= window_size:
                        # 我们希望最终的拼接长度是 window_size。
                        # 所以，从 past_k 中保留 window_size - k_current_step.shape[2] 个元素。
                        # 这些元素从 past_k 的末尾取。
                        num_elements_to_keep_from_past = window_size - k_current_step.shape[2]
                        
                        if num_elements_to_keep_from_past < 0:
                            # 这意味着当前步本身就比窗口大，只保留当前步（并可能裁剪当前步）
                            # 但在典型的 step 模式下 k_current_step.shape[2] 是 1，所以这里不应该发生
                            print(f"Warning: k_current_step ({k_current_step.shape[2]}) >= window_size ({window_size}). Truncating current step.")
                            k_to_concat_from_past = torch.empty_like(past_k[:,:,0:0,:]) # Empty tensor with correct dims
                            v_to_concat_from_past = torch.empty_like(past_v[:,:,0:0,:])
                            k_current_step = k_current_step[:, :, :window_size, :]
                            v_current_step = v_current_step[:, :, :window_size, :]
                        elif num_elements_to_keep_from_past == 0:
                             k_to_concat_from_past = torch.empty_like(past_k[:,:,0:0,:])
                             v_to_concat_from_past = torch.empty_like(past_v[:,:,0:0,:])
                        else:
                            k_to_concat_from_past = past_k[:, :, -num_elements_to_keep_from_past:]
                            v_to_concat_from_past = past_v[:, :, -num_elements_to_keep_from_past:]
                        
                        k = torch.cat((k_to_concat_from_past, k_current_step), dim=2)
                        v = torch.cat((v_to_concat_from_past, v_current_step), dim=2)

                    elif past_k.shape[2] + k_current_step.shape[2] > window_size:
                        # 过去缓存未满，但加上当前步就满了/超了，也需要裁剪
                        num_elements_to_keep_from_past = window_size - k_current_step.shape[2]
                        if num_elements_to_keep_from_past < 0: num_elements_to_keep_from_past = 0 # Should not happen here

                        if num_elements_to_keep_from_past <= past_k.shape[2]:
                             k_to_concat_from_past = past_k[:, :, -num_elements_to_keep_from_past:]
                             v_to_concat_from_past = past_v[:, :, -num_elements_to_keep_from_past:]
                        else: # num_to_keep_from_past > past_k.shape[2], means keep all of past_k
                             k_to_concat_from_past = past_k
                             v_to_concat_from_past = past_v
                        
                        k = torch.cat((k_to_concat_from_past, k_current_step), dim=2)
                        v = torch.cat((v_to_concat_from_past, v_current_step), dim=2)
                    else:
                        # 缓存未满，并且加上当前步也不会超，直接拼接
                        k = torch.cat((past_k, k_current_step), dim=2)
                        v = torch.cat((past_v, v_current_step), dim=2)
                else:
                    # 没有窗口限制，直接拼接
                    k = torch.cat((past_k, k_current_step), dim=2)
                    v = torch.cat((past_v, v_current_step), dim=2)
            else:
                 print("Warning: past_kv contained invalid types. Skipping cache concatenation.")
                 print("警告: past_kv 包含无效类型。跳过缓存拼接。")
        present_kv: Optional[Tuple[Tensor, Tensor]] = (k,v) # 当前的 K, V 状态，用于下一时间步的缓存
        
        # effective_causal 确定是否在 scaled_dot_product_attention 中使用 is_causal 参数
        effective_causal = causal and past_kv is None and attn_mask is None
        sdpa_attn_mask = attn_mask # scaled_dot_product_attention 使用的注意力掩码
        
        # 使用 PyTorch 内置的 scaled_dot_product_attention 进行高效计算
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_attn_mask, # 传入注意力掩码
            is_causal=effective_causal, # 如果满足条件，则启用内置的因果掩码
            dropout_p=self.dropout.p if self.training else 0.0 # 训练时应用 dropout
        )
        # 将注意力输出转置并重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output) # 通过输出投影层
        return output, present_kv # 返回输出和当前的 K, V 缓存

# --- Jamba Block (copied, uses JambaSConfig) ---
class JambaBlock(nn.Module):
    """
    Jamba 模型的基本构建块。
    每个块可以是一个 Mamba 层或一个注意力层，具体取决于配置。
    """
    def __init__(self, layer_idx: int, config: JambaSConfig, device=None): # Uses JambaSConfig, 使用 JambaSConfig
        super().__init__()
        self.layer_idx = layer_idx # 当前层的索引
        self.config = config
        # 判断当前层是否为注意力层
        self.is_attention_layer = layer_idx in (config.use_attention_layer_indices or [])
        self.norm = RMSNorm(config.d_model, eps=1e-5, device=device) # RMSNorm 归一化层
        if self.is_attention_layer: # 如果是注意力层
            self.mixer = MultiHeadAttention(config, device=device) # 使用多头注意力
        else: # 否则
            self.mixer = Mamba2(config, device=device) # 使用 Mamba2 层

    def forward(self, x: Tensor, layer_cache: Optional[Tuple[Optional[MambaInferenceCache], Optional[Tuple[Tensor, Tensor]]]] = None, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tuple[Optional[MambaInferenceCache], Optional[Tuple[Tensor, Tensor]]]]:
        """
        前向传播函数。
        Args:
            x: 输入张量。
            layer_cache: 可选的层缓存，包含 Mamba 缓存和注意力 K,V 缓存。
            attn_mask: 可选的注意力掩码。
        Returns:
            输出张量和更新后的层缓存。
        """
        residual = x # 残差连接
        x_norm = self.norm(x) # 归一化
        mamba_cache_in, attn_kv_cache_in = layer_cache if layer_cache is not None else (None, None) # 解包输入缓存
        mamba_cache_out, attn_kv_cache_out = None, None # 初始化输出缓存
        
        if self.is_attention_layer: # 如果是注意力层
            attn_output, attn_kv_cache_out = self.mixer(x_norm, past_kv=attn_kv_cache_in, causal=True, attn_mask=attn_mask) # 计算注意力输出
            if attn_output is not None: x = residual + attn_output # 应用残差连接
            else: x = residual # 如果注意力输出为 None (例如 num_heads=0)，则仅残差
            mamba_cache_out = mamba_cache_in # Mamba 缓存不变
        else: # 如果是 Mamba 层
            mamba_output, mamba_cache_out = self.mixer(x_norm, h=mamba_cache_in) # 计算 Mamba 输出
            x = residual + mamba_output # 应用残差连接
            attn_kv_cache_out = attn_kv_cache_in # 注意力缓存不变
        return x, (mamba_cache_out, attn_kv_cache_out) # 返回输出和更新后的缓存

    def step(self, x_step: Tensor, layer_cache: Tuple[Optional[MambaInferenceCache], Optional[Tuple[Tensor, Tensor]]]) -> Tuple[Tensor, Tuple[Optional[MambaInferenceCache], Optional[Tuple[Tensor, Tensor]]]]:
        """
        单步前向传播函数，用于自回归生成。
        Args:
            x_step: 当前时间步的输入张量。
            layer_cache: 当前层的缓存。
        Returns:
            当前时间步的输出张量和更新后的层缓存。
        """
        residual = x_step # 残差连接
        x_norm = self.norm(x_step) # 归一化
        mamba_cache_in, attn_kv_cache_in = layer_cache # 解包输入缓存
        mamba_cache_out, attn_kv_cache_out = mamba_cache_in, attn_kv_cache_in # 初始化输出缓存 (先设为输入缓存)
        
        if self.is_attention_layer: # 如果是注意力层
             # 注意: 在 step 模式下，causal=False，因为我们只处理一个时间步，因果性由 K,V 缓存处理
             attn_output, attn_kv_cache_out = self.mixer(x_norm, past_kv=attn_kv_cache_in, causal=False, attn_mask=None)
             if attn_output is not None: x_step = residual + attn_output # 应用残差连接
             else: x_step = residual
        else: # 如果是 Mamba 层
             if mamba_cache_in is None: # 如果 Mamba 缓存未初始化
                 batch_size = x_step.shape[0]
                 current_device = x_step.device
                 # 为 Mamba 推理分配缓存
                 mamba_cache_in = MambaInferenceCache.alloc(batch_size, self.config, device=current_device)
             mamba_output, mamba_cache_out = self.mixer.step(x_norm, h=mamba_cache_in) # 执行 Mamba 的单步操作
             x_step = residual + mamba_output # 应用残差连接
        return x_step, (mamba_cache_out, attn_kv_cache_out) # 返回输出和更新后的缓存

# Define LayerCache type alias for this new file
# 为此新文件定义 LayerCacheS 类型别名
LayerCacheS: TypeAlias = Tuple[Optional[MambaInferenceCache], Optional[Tuple[Tensor, Tensor]]]


# --- Main Jamba Model for S-in S-out Audio Generation ---
class JambaForAudioGeneration_S_In_S_Out(nn.Module):
    """
    Jamba 模型，用于从 S 码元序列输入生成 S 码元序列输出的音频任务。
    集成了风格编码器，允许模型根据参考音频的风格进行生成。
    """
    def __init__(self, config: JambaSConfig, device=None):
        super().__init__()
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Style Encoder (using s_indices)
        # 1. 风格编码器 (使用 s_indices)
        self.style_encoder = StyleEncoderUsingS(config, device=self.device)

        # 2. Input s-indices Embeddings (for main model)
        # 2. 输入 s-indices 嵌入层 (用于主模型)
        self.s_embeddings = nn.ModuleList([
            nn.Embedding(config.dac_codebook_size, config.s_embedding_dim_individual, device=self.device)
            for _ in range(config.dac_num_quantizers)
        ])
        
        # 计算组合后的 s 嵌入维度
        combined_s_embedding_dim = config.dac_num_quantizers * config.s_embedding_dim_individual
        # 线性投影层，将组合的 s 嵌入投影到 d_model 维度
        self.s_input_projection = nn.Linear(
            combined_s_embedding_dim,
            config.d_model,
            bias=False,
            device=self.device
        )
        
        # 3. Style Vector Projection (to d_model for addition)
        # 3. 风格向量投影 (投影到 d_model 以便与主干网络特征相加)
        self.style_proj_for_addition = nn.Linear(
            config.style_dim,
            config.d_model,
            bias=False, # Typically bias is false for projections that are added
                        # 通常用于相加的投影层，偏置设为 False
            device=self.device
        )

        # 4. Jamba Layers
        # 4. Jamba 层堆叠
        self.layers = nn.ModuleList(
            [JambaBlock(i, config, device=self.device) for i in range(config.n_layer)]
        )

        # 5. Final Norm
        # 5. 最终的归一化层
        self.norm_f = RMSNorm(config.d_model, eps=1e-5, device=self.device)

        # 6. Output Projection to predict next s_indices logits
        # 6. 输出投影层，用于预测下一个 s_indices 的 logits
        self.output_proj = nn.Linear(
            config.d_model,
            config.dac_num_quantizers * config.dac_codebook_size, # 输出维度是 量化器数量 * 码本大小
            bias=False,
            device=self.device
        )
        
        print(f"Initialized JambaForAudioGeneration_S_In_S_Out on device {self.device}")
        print(f"在设备 {self.device} 上初始化 JambaForAudioGeneration_S_In_S_Out")
        print(f"  Main s-emb dim (individual): {config.s_embedding_dim_individual}")
        print(f"  主模型 s-emb 维度 (单个): {config.s_embedding_dim_individual}")
        # ... other prints ...
        # ... 其他打印信息 ...
        attn_indices = config.use_attention_layer_indices if isinstance(config.use_attention_layer_indices, list) else []
        print(f"  Using Attention for layers: {attn_indices}")
        print(f"  使用注意力机制的层: {attn_indices}")

        self.apply(self._initialize_weights) # 应用权重初始化

    def _initialize_weights(self, module):
        """
        初始化模型权重。
        """
        if isinstance(module, (nn.Linear, nn.Conv1d)): # 对线性和卷积层使用 Xavier 均匀初始化
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: # 如果有偏置，则初始化为0
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): # 对嵌入层使用正态分布初始化
            torch.nn.init.normal_(module.weight, std=0.02)

    def get_style_encoder(self) -> StyleEncoderUsingS: # Return type updated, 返回类型已更新
        """
        获取风格编码器实例。
        """
        return self.style_encoder

    def forward(self, 
                s_indices_input: Tensor, # 形状: (B, T, Q) - 用于主模型的离散码本索引
                s_originals_for_style: List[Tensor], # StyleEncoder 使用的 (T_orig_style, Q) 列表
                lengths_for_style: Tensor, # 对应 s_originals_for_style 的长度张量 (B,)
                attn_mask: Optional[Tensor] = None, # 可选的注意力掩码
                past_caches: Optional[List[LayerCacheS]] = None # 可选的层缓存列表
               ) -> Tuple[Tensor, List[LayerCacheS]]:
        """
        前向传播函数。
        Args:
            s_indices_input: 主自回归部分的输入 s_indices (B, T, Q)。
            s_originals_for_style: 用于风格提取的未填充 s_indices 列表。每个元素形状 (T_orig_style, Q)。
            lengths_for_style: s_originals_for_style 中序列的长度 (B,)。
            attn_mask: s_indices_input 的注意力掩码 (B, 1, 1, T)。
            past_caches: 可选的层缓存列表。
        Returns:
            元组，包含 (logits (B, T, Q, C)，更新后的缓存列表)。
        """
        batch_size, original_seq_len, num_quantizers_q = s_indices_input.shape # 重命名 seq_len 为 original_seq_len
        assert num_quantizers_q == self.config.dac_num_quantizers, (
            f"Input s_indices_input last dim ({num_quantizers_q}) != "
            f"config.dac_num_quantizers ({self.config.dac_num_quantizers})"
        ) # 检查输入 s_indices 的量化器数量是否与配置一致
        
        s_indices_input = s_indices_input.to(self.device) # 将输入移至设备
        lengths_for_style = lengths_for_style.to(self.style_encoder.device) # 确保长度张量在风格编码器的设备上
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device) # 将注意力掩码移至设备

        # --- START: Model Internal Padding for Mamba layers ---
        current_seq_len_for_model = original_seq_len
        input_was_padded_internally = False

        is_any_mamba_layer = any(not (idx in (self.config.use_attention_layer_indices or [])) for idx in range(self.config.n_layer))

        if is_any_mamba_layer and self.config.chunk_size > 0:
            if original_seq_len % self.config.chunk_size != 0:
                target_seq_len = ((original_seq_len - 1) // self.config.chunk_size + 1) * self.config.chunk_size
                num_padding = target_seq_len - original_seq_len
                
                # Pad s_indices_input. 使用 (dac_codebook_size - 1) 作为填充值 (EOS token).
                pad_value_for_s_input = self.config.dac_codebook_size - 1
                s_indices_input = F.pad(s_indices_input, (0,0, 0,num_padding, 0,0), mode='constant', value=pad_value_for_s_input)
                # print(f"DEBUG: Model internal padding: Padded s_indices_input from {original_seq_len} to {target_seq_len}")
                
                if attn_mask is not None:
                    # attn_mask 预期形状 (B, 1, 1, T_orig) from collate_fn
                    if attn_mask.shape[-1] == original_seq_len:
                         attn_mask = F.pad(attn_mask, (0, num_padding), mode='constant', value=True) # Pad last dim with True
                         # print(f"DEBUG: Model internal padding: Padded attn_mask to target_seq_len {target_seq_len}")
                    # else:
                         # print(f"DEBUG: Warning: attn_mask shape {attn_mask.shape} last dim doesn't match original_seq_len {original_seq_len}. Not padding attn_mask.")
                
                current_seq_len_for_model = target_seq_len
                input_was_padded_internally = True
        # --- END: Model Internal Padding ---

        # 1. Calculate Style Vector
        style_vector = self.style_encoder(s_originals_for_style, lengths_for_style) # (B, style_dim)
        style_vector = style_vector.to(self.device) # 确保风格向量在主模型的设备上

        # 2. Embed s_indices (主输入) 并投影到 d_model
        embeddings_list = [] # 存储每个量化器的嵌入结果
        for q_idx in range(self.config.dac_num_quantizers):
            embeddings_list.append(self.s_embeddings[q_idx](s_indices_input[:, :, q_idx]))
        
        x = torch.cat(embeddings_list, dim=-1) # 沿最后一个维度拼接嵌入结果
        x = self.s_input_projection(x) # (B, T_padded_or_orig, d_model)，投影到 d_model

        # 3. Inject Style Vector
        projected_style = self.style_proj_for_addition(style_vector) # (B, d_model)，投影风格向量
        x = x + projected_style.unsqueeze(1) # 将风格向量加到每个时间步 (通过广播机制)

        # 4. Initialize caches if not provided
        if past_caches is None or len(past_caches) != self.config.n_layer:
            if past_caches is not None: # 如果提供了无效的缓存结构
                 print("警告: 提供的 past_caches 结构无效，正在重新初始化。")
            past_caches = []
            attn_indices_cfg = self.config.use_attention_layer_indices or [] # 获取注意力层索引
            for i in range(self.config.n_layer): # 遍历每一层
                 is_attn = i in attn_indices_cfg # 判断是否为注意力层
                 mamba_c = MambaInferenceCache.alloc(batch_size, self.config, device=self.device) if not is_attn else None #向Mamba 推理缓存传递当前序列长度
                 attn_c = None # 注意力缓存初始为 None
                 past_caches.append((mamba_c, attn_c))
        
        new_caches: List[LayerCacheS] = [] # 用于存储新的缓存

        # 5. Pass through Jamba Blocks
        for i, layer in enumerate(self.layers): # 遍历每一层
            layer_cache_in = past_caches[i] # 获取当前层的输入缓存
            x, layer_cache_out = layer(x, layer_cache=layer_cache_in, attn_mask=attn_mask) # 前向传播
            new_caches.append(layer_cache_out) # 存储输出缓存

        # 6. Final Norm
        x = self.norm_f(x)

        # 7. Output Projection
        logits = self.output_proj(x) # (B, T_padded_or_orig, Q*C)
        
        # --- START: Model Internal Unpadding of Logits ---
        if input_was_padded_internally and original_seq_len < current_seq_len_for_model:
            logits = logits[:, :original_seq_len, :] # Unpad sequence dimension
            # print(f"DEBUG: Model internal unpadding: Unpadded logits from {current_seq_len_for_model} back to {original_seq_len}")
        # --- END: Model Internal Unpadding of Logits ---
        
        logits = rearrange(logits, 'b t (q c) -> b t q c', 
                           q=self.config.dac_num_quantizers, 
                           c=self.config.dac_codebook_size) # (B, T_orig, Q, C)
        
        return logits, new_caches

    def step(self, 
             s_step_indices: Tensor, # Shape: (B, Q) or (B, 1, Q)
                                     # 形状: (B, Q) 或 (B, 1, Q)
             caches: List[LayerCacheS], # 当前所有层的缓存列表
             style_vector: Tensor # Pre-computed style vector (B, style_dim)
                                  # 预先计算的风格向量 (B, style_dim)
            ) -> Tuple[Tensor, List[LayerCacheS]]:
        """
        Single step for autoregressive generation with style vector.
        使用风格向量进行自回归生成的单步操作。
        """
        s_step_indices = s_step_indices.to(self.device) # 将单步输入移至设备
        style_vector = style_vector.to(self.device) # Ensure style vector is on correct device
                                                    # 确保风格向量在正确的设备上

        if s_step_indices.ndim == 2: # 如果输入是 (B, Q)
            s_step_indices = s_step_indices.unsqueeze(1) # 扩展为 (B, 1, Q)
        
        batch_size = s_step_indices.shape[0]
        assert s_step_indices.shape[2] == self.config.dac_num_quantizers, "s_step_indices incorrect Q dim" # 检查 Q 维度
        assert s_step_indices.shape[1] == 1, "s_step_indices should be for a single time step" # 确保是单个时间步

        # 1. Embed s_indices and project
        # 1. 嵌入 s_indices 并投影
        embeddings_list_step = [] # 存储单步的嵌入结果
        for q_idx in range(self.config.dac_num_quantizers):
            # s_step_indices[:, 0, q_idx] 是 (B,)
            embeddings_list_step.append(self.s_embeddings[q_idx](s_step_indices[:, 0, q_idx])) 
        
        x_step_emb = torch.cat(embeddings_list_step, dim=-1) # (B, Q * s_emb_dim_individual)
        x_step = self.s_input_projection(x_step_emb).unsqueeze(1) # (B, 1, d_model)
        
        # 2. Inject Style Vector
        # 2. 注入风格向量
        projected_style_step = self.style_proj_for_addition(style_vector) # (B, d_model)
        x_step = x_step + projected_style_step.unsqueeze(1) # (B, 1, d_model)


        # 3. Initialize caches if necessary (should generally be passed correctly)
        # 3. 如有必要，初始化缓存 (通常应该正确传递)
        if not caches or len(caches) != self.config.n_layer:
             print("Warning: Invalid or empty caches in step function. Re-initializing.")
             print("警告: step 函数中的缓存无效或为空。正在重新初始化。")
             caches = []
             attn_indices_cfg = self.config.use_attention_layer_indices or []
             for i in range(self.config.n_layer):
                 is_attn = i in attn_indices_cfg
                 mamba_c = MambaInferenceCache.alloc(batch_size, self.config, device=self.device) if not is_attn else None
                 attn_c = None
                 caches.append((mamba_c, attn_c))

        new_caches_step: List[LayerCacheS] = [] # 存储单步操作后的新缓存
        
        # 4. Pass through Jamba Blocks
        # 4. 通过 Jamba 模块
        for i, layer in enumerate(self.layers): # 遍历每一层
            layer_cache_in = caches[i] # 获取当前层的输入缓存
            x_step, layer_cache_out = layer.step(x_step, layer_cache=layer_cache_in) # 执行单步操作
            new_caches_step.append(layer_cache_out) # 存储输出缓存
            
        # 5. Final Norm
        # 5. 最终归一化
        x_step = self.norm_f(x_step)

        # 6. Output Projection
        # 6. 输出投影
        logits_step = self.output_proj(x_step) # (B, 1, Q*C)
        logits_step = rearrange(logits_step, 'b t (q c) -> b t q c', # t 仍然是 1
                                q=self.config.dac_num_quantizers,
                                c=self.config.dac_codebook_size) # (B, 1, Q, C)
                                
        return logits_step, new_caches_step # 返回单步 logits 和新的缓存

    def save_pretrained(self, save_directory: str):
        """
        保存预训练模型和配置文件。
        Args:
            save_directory: 保存模型和配置的目录。
        """
        import json # Ensure json is imported for saving, 确保导入 json 以便保存
        if not os.path.exists(save_directory): # 如果目录不存在，则创建
            os.makedirs(save_directory)
        config_path = os.path.join(save_directory, "config_s_in_s_out_styled.json") # New name for styled config, 带风格配置的新名称
        
        # 将配置保存为 JSON 文件
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=4)

        model_path = os.path.join(save_directory, "pytorch_model_s_in_s_out_styled.bin") # 带风格模型的权重文件名
        torch.save(self.state_dict(), model_path) # 保存模型权重
        print(f"S-in-S-out STYLED model config and weights saved to {save_directory}")
        print(f"S输入S输出 (带风格) 模型的配置和权重已保存到 {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, device=None):
        """
        从预训练路径加载模型和配置。
        会尝试加载不同命名约定的配置文件和模型权重，以兼容旧版本。
        Args:
            pretrained_model_name_or_path: 包含模型和配置的目录路径。
            device: 加载模型的设备。
        Returns:
            加载的 JambaForAudioGeneration_S_In_S_Out 模型实例。
        """
        import json # Ensure json is imported for loading, 确保导入 json 以便加载
        import dataclasses # 用于检查 dataclass 字段的 MISSING 属性

        # Try to load specific filenames for the styled version
        # 尝试加载带风格版本的特定文件名
        config_path = os.path.join(pretrained_model_name_or_path, "config_s_in_s_out_styled.json")
        model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model_s_in_s_out_styled.bin")

        # If styled version filenames don't exist, try falling back to non-styled S->S version filenames
        # 如果带风格版本的文件名不存在，则尝试回退到不带风格的S->S版本文件名
        if not os.path.exists(config_path):
            config_path_old = os.path.join(pretrained_model_name_or_path, "config_s_in_s_out.json")
            if os.path.exists(config_path_old) : config_path = config_path_old
        if not os.path.exists(model_path):
            model_path_old = os.path.join(pretrained_model_name_or_path, "pytorch_model_s_in_s_out.bin")
            if os.path.exists(model_path_old) : model_path = model_path_old
        
        # If neither of the above exist, try the most original (non S->S) filenames as a last resort
        # 如果上述两种文件名都不存在，则尝试最原始的 (非S->S) 文件名作为最后手段
        if not os.path.exists(config_path) or not os.path.exists(model_path):
             original_config_path = os.path.join(pretrained_model_name_or_path, "config.json")
             original_model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
             if os.path.exists(original_config_path) and os.path.exists(original_model_path):
                 print(f"警告: 未找到S->S特定配置文件。尝试加载原始配置文件 '{original_config_path}' 并用 JambaSConfig 解析。")
                 config_path = original_config_path
                 model_path = original_model_path
             else:
                 raise FileNotFoundError(f"在路径 '{pretrained_model_name_or_path}' 中找不到模型配置文件或权重文件。"
                                         f"(尝试了 '..._styled.json', '..._s_in_s_out.json', 和 'config.json')")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f) # 加载配置字典
        
        # Ensure the loaded config_dict contains all fields required by JambaSConfig,
        # populating missing style-related fields with defaults from JambaSConfig if loading an older config.
        # 确保加载的 config_dict 包含 JambaSConfig 所需的所有字段，
        # 如果加载的是旧版配置，则使用 JambaSConfig 中的默认值填充缺失的风格相关字段。
        config_fields = JambaSConfig.__dataclass_fields__ # 获取 JambaSConfig 的所有字段定义
        for field_name, field_def in config_fields.items(): # 遍历每个字段
            if field_name not in config_dict: # 如果加载的配置中缺少该字段
                # If the field has a simple default value (not a factory function)
                # 如果字段有简单默认值 (非工厂函数)
                if field_def.default is not dataclasses.MISSING and field_def.default is not field_def.default_factory:
                    config_dict[field_name] = field_def.default
                # If the field has a default factory function
                # 如果字段有默认工厂函数
                elif field_def.default_factory is not dataclasses.MISSING and field_def.default_factory is not field_def.default_factory: # The latter check ensures it's truly a factory
                                                                                                                                   # 后一个判断是确保它真的是一个工厂
                    config_dict[field_name] = field_def.default_factory()
                # Special handling for style-related parameters, if old config doesn't have them, JambaSConfig defaults will be used
                # 特别处理风格相关参数，如果旧config没有这些，JambaSConfig的默认值会被使用
                elif field_name in ['style_dim', 'style_s_embedding_dim_individual', 
                                    'style_encoder_num_conv_layers', 'style_encoder_kernel_size']:
                    print(f"信息: 加载的配置文件中缺少风格相关参数 '{field_name}'。将使用 JambaSConfig 中的默认值。")
                    # JambaSConfig(**config_dict) will automatically use these fields' default values
                    # JambaSConfig(**config_dict) 会自动使用这些字段的默认值
                # else: # For other necessary fields without defaults, missing them might cause errors
                      # 对于其他没有默认值的必要字段，如果缺失可能会导致错误
                    # print(f"调试: 字段 {field_name} 在加载的字典中不存在，并且没有简单的默认值。")


        config = JambaSConfig(**config_dict) # Use loaded (and potentially populated) dict to create config object
                                             # 使用加载（并可能已填充）的字典创建配置对象

        if device is None: # 如果未指定设备，则自动选择
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = cls(config, device=device) # Create model instance with config and device
                                           # 使用配置和设备创建模型实例
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)) # Load model weights
                                                                                              # 加载模型权重
        model.eval() # Set model to evaluation mode, 将模型设置为评估模式
        print(f"S输入S输出 (带风格) 模型已从 '{pretrained_model_name_or_path}' 加载 (使用配置文件: '{config_path}')")
        return model

# --- Helper: Collate Function for S-Input S-Output with Style ---
def jamba_s_styled_collate_fn(batch: List[Tensor], 
                              pad_value_s: int = -100, # S 码元填充值
                              eos_token_id: Optional[int] = None, # EOS 标记 ID
                             # chunk_size_for_padding: int = 0 # 已移除，模型内部处理
                             ) -> Tuple[Tensor, Tensor, Optional[Tensor], List[Tensor], Tensor]: 
    """
    Collate function for JambaForAudioGeneration_S_In_S_Out with StyleEncoder.
    用于带风格编码器的 JambaForAudioGeneration_S_In_S_Out 模型的 Collate 函数。
    Args:
        batch: s_indices 张量列表，每个形状为 (T_orig, Q)。这些既用于风格提取，也作为模型输入/目标的基础。
        pad_value_s: 填充值 (用于目标填充)。
        eos_token_id: EOS 标记 ID (用于输入填充和目标构建)。
    Returns:
        s_model_input_padded: (B, T_padded_input, Q) - 填充后的模型输入 S 码元
        s_target_padded: (B, T_padded_target, Q) - 填充后的目标 S 码元
        attn_mask: (B, 1, 1, T_padded_input) - 注意力掩码
        s_originals_for_style: List of original (T_orig, Q) tensors (effectively the input `batch`)
                               用于风格编码器的原始 (T_orig, Q) 张量列表 (实际上就是输入的 `batch`)
        lengths_for_style: (B,) tensor of original lengths for style encoder.
                           用于风格编码器的原始长度张量 (B,)。
    """
    s_originals_for_style = batch # 这个列表用于风格编码器
    
    original_lengths_for_style = torch.tensor([s.shape[0] for s in s_originals_for_style], dtype=torch.long)

    s_model_input_list = [] # 模型输入列表
    s_target_list = [] # 目标列表
    processed_lengths_for_mask = [] # 处理后的长度，用于创建注意力掩码

    for s_orig in s_originals_for_style: 
        if eos_token_id is not None: 
            eos_tensor = torch.full((1, s_orig.shape[1]), eos_token_id, 
                                    dtype=s_orig.dtype, device=s_orig.device)
            s_complete_target = torch.cat((s_orig, eos_tensor), dim=0) 
        else:
            s_complete_target = s_orig 

        s_input_for_model = s_complete_target[:-1, :] 
        s_target_for_loss = s_complete_target[1:, :]  

        processed_lengths_for_mask.append(s_input_for_model.shape[0]) 
        s_model_input_list.append(s_input_for_model)
        s_target_list.append(s_target_for_loss)

    max_len_input = 0 
    if processed_lengths_for_mask: 
        max_len_input = max(l for l in processed_lengths_for_mask) 

    # 移除 chunk_size_for_padding 相关逻辑
    # if chunk_size_for_padding > 0 and max_len_input > 0:
    #     if max_len_input % chunk_size_for_padding != 0:
    #         max_len_input = (max_len_input // chunk_size_for_padding + 1) * chunk_size_for_padding

    s_model_input_padded_list = [] 
    s_target_padded_list = [] 
    attn_mask_list = [] 

    for s_in, s_targ, in_len in zip(s_model_input_list, s_target_list, processed_lengths_for_mask):
        num_padding_in = max_len_input - in_len 
        # 使用 eos_token_id (一个有效的token ID) 来填充输入，而不是 pad_value_s (-100)
        # 这是正确的，因为模型输入不应包含用于损失忽略的 -100
        if eos_token_id is None:
            raise ValueError("eos_token_id must be provided to jamba_s_styled_collate_fn for input padding.")
        s_in_padded = F.pad(s_in, (0, 0, 0, num_padding_in), mode='constant', value=eos_token_id)
        s_model_input_padded_list.append(s_in_padded)
        
        mask = torch.zeros(1, 1, 1, max_len_input, dtype=torch.bool, device=s_in.device) # (B, H, T_q, T_k) -> (1,1,1,max_len_input) for causal
        if num_padding_in > 0:
            mask[:, :, :, -num_padding_in:] = True 
        attn_mask_list.append(mask)

        num_padding_targ = max_len_input - s_targ.shape[0] 
        s_targ_padded = F.pad(s_targ, (0, 0, 0, num_padding_targ), mode='constant', value=pad_value_s) # 目标使用 pad_value_s
        s_target_padded_list.append(s_targ_padded)

    s_model_input_final_padded = torch.stack(s_model_input_padded_list, dim=0) if s_model_input_padded_list else torch.empty(0)
    s_target_final_padded = torch.stack(s_target_padded_list, dim=0) if s_target_padded_list else torch.empty(0)
    attn_mask_final = torch.cat(attn_mask_list, dim=0) if attn_mask_list else None #  (B, 1, 1, T_padded_input)
    
    return s_model_input_final_padded, s_target_final_padded, attn_mask_final, s_originals_for_style, original_lengths_for_style


# --- Example Usage (Conceptual) ---
if __name__ == '__main__':
    import json # Ensure json is imported for main example if needed by save/load
                # 如果保存/加载需要，确保在主示例中导入 json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running S-in-S-out STYLED example on device {device}")
    print(f"在设备 {device} 上运行 S输入S输出 (带风格) 示例")

    EOS_TOKEN = 1024 # 定义 EOS 标记的值

    # 配置 JambaSConfig 实例
    config = JambaSConfig(
        d_model=768,
        n_layer=8,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        chunk_size=128,
        num_attention_heads=8,
        attention_dropout=0.1,
        dac_codebook_size= EOS_TOKEN + 1, # 码本大小，包括 EOS
        dac_num_quantizers=9, # 量化器数量
        s_embedding_dim_individual=64, # 主模型 S 码元嵌入维度
        use_attention_layer_indices=[1,3,5,7], # 指定使用注意力的层
        # Style specific
        # 风格特定参数
        style_dim=64, # 风格向量维度
        style_s_embedding_dim_individual=32, # Smaller for style, 风格编码器中 S 码元嵌入维度 (可以更小)
        style_encoder_num_conv_layers=4, # 风格编码器卷积层数
        style_encoder_kernel_size=5 # 风格编码器卷积核大小
    )
    print(f"Using S-Config (Styled): {asdict(config)}")
    print(f"使用 S-Config (带风格): {asdict(config)}")

    # 创建模型实例
    model = JambaForAudioGeneration_S_In_S_Out(config, device=device)
    # 计算可训练参数数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_style_encoder = sum(p.numel() for p in model.style_encoder.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params / 1e6:.2f} M")
    print(f"总可训练参数: {n_params / 1e6:.2f} M")
    print(f"  Style Encoder parameters: {n_params_style_encoder / 1e6:.2f} M")
    print(f"  风格编码器参数: {n_params_style_encoder / 1e6:.2f} M")


    # --- 准备虚拟数据 ---
    num_samples = 10 # 样本数量
    batch_size = 2 # 批次大小
    min_len, max_len = 90, 120 # Longer sequences for style, 风格序列使用较长长度
    dummy_s_sequences = [] # 存储虚拟 S 码元序列的列表
    for _ in range(num_samples):
        original_len = torch.randint(min_len, max_len + 1, (1,)).item() # 随机生成原始长度
        # 随机生成 S 码元索引
        s_indices = torch.randint(0, config.dac_codebook_size -1, (original_len, config.dac_num_quantizers), dtype=torch.long)
        dummy_s_sequences.append(s_indices)

    # Use new collate_fn
    # 使用新的 collate_fn
    collate_fn_s_styled = partial(jamba_s_styled_collate_fn, pad_value_s=-100, eos_token_id=EOS_TOKEN) # 移除 chunk_size_for_padding
    # 创建 DataLoader
    dataloader = torch.utils.data.DataLoader(dummy_s_sequences, batch_size=batch_size, collate_fn=collate_fn_s_styled)
    
    print(f"Created DataLoader with {len(dataloader)} batches.")
    print(f"创建了包含 {len(dataloader)} 个批次的 DataLoader。")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # AdamW 优化器
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # 交叉熵损失函数，忽略填充值 -100
    
    # --- 虚拟训练循环 ---
    model.train() # 设置模型为训练模式
    print("Starting dummy training loop (with style encoder)...")
    print("开始虚拟训练循环 (带风格编码器)...")
    for epoch in range(1): # Single epoch for brevity, 为简洁起见，只训练一个 epoch
        for s_model_input, s_target, attn_mask_for_loss, s_originals_style, lengths_style in dataloader:
            # 将数据移至设备
            s_model_input = s_model_input.to(device)
            s_target = s_target.to(device)
            # s_originals_style items are moved to device inside StyleEncoder / main forward
            # s_originals_style 的元素会在 StyleEncoder / 主模型 forward 内部移至设备
            lengths_style = lengths_style.to(device) # ensure on device for StyleEncoder call
                                                     # 确保在 StyleEncoder 调用时位于设备上
            if attn_mask_for_loss is not None:
                attn_mask_for_loss = attn_mask_for_loss.to(device)

            optimizer.zero_grad() # 清空梯度
            # 模型前向传播
            logits, _ = model(s_indices_input=s_model_input, 
                               s_originals_for_style=s_originals_style,
                               lengths_for_style=lengths_style,
                               attn_mask=attn_mask_for_loss) 
            
            # 计算损失
            # logits: (B, T, Q, C), s_target: (B, T, Q)
            # reshape logits to (B*T*Q, C) and s_target to (B*T*Q)
            loss = loss_fn(logits.reshape(-1, config.dac_codebook_size), 
                           s_target.reshape(-1))
            
            if torch.isnan(loss): # 检查损失是否为 NaN
                print("NaN loss detected, skipping batch.")
                continue
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            print(f"Epoch {epoch+1}, Batch Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} finished.")
        print(f"Epoch {epoch+1} 已完成。")

    # --- 自回归生成示例 (带风格，EOS 停止) ---
    print("\n--- Autoregressive Generation Example (STYLED, with EOS stopping) ---")
    print("\n--- 自回归生成示例 (带风格，EOS 停止) ---")
    model.eval() # 设置模型为评估模式
    
    # 1. Prepare prompt for style extraction (e.g., 100 steps)
    #    This prompt is ALSO the beginning of the generation sequence.
    # 1. 准备用于风格提取的提示 (例如，100 个时间步)
    #    此提示也是生成序列的开头。
    style_prompt_s_indices = dummy_s_sequences[0][:100, :].to(device) # (100, Q)
    print(f"Using style prompt of shape: {style_prompt_s_indices.shape}")
    print(f"使用形状为 {style_prompt_s_indices.shape} 的风格提示")

    # For StyleEncoder, it needs a list of tensors and lengths
    # StyleEncoder 需要一个张量列表和长度列表
    style_prompt_list = [style_prompt_s_indices] # 包含单个风格提示的列表
    style_lengths = torch.tensor([style_prompt_s_indices.shape[0]], dtype=torch.long).to(model.style_encoder.device) # 风格提示的长度

    # 预计算风格向量
    with torch.no_grad(): # 不计算梯度进行生成
        precomputed_style_vector = model.get_style_encoder()(style_prompt_list, style_lengths) # (1, style_dim)
        precomputed_style_vector = precomputed_style_vector.to(device) # Ensure on main model device
                                                                        # 确保在主模型设备上
    print(f"Precomputed style vector shape: {precomputed_style_vector.shape}")
    print(f"预计算的风格向量形状: {precomputed_style_vector.shape}")

    # 2. Initialize KV caches using the same style prompt (or part of it)
    #    The part of the prompt used here will be the actual starting tokens for generation.
    #    Let's use the last few tokens of the style_prompt as the generation_prompt.
    # 2. 使用相同的风格提示 (或其一部分) 初始化 KV 缓存
    #    此处使用的提示部分将是生成的实际起始标记。
    #    让我们使用 style_prompt 的最后几个标记作为 generation_prompt。
    generation_prompt_len = 5 # 生成提示的长度
    generation_prompt_s_indices = style_prompt_s_indices[-generation_prompt_len:, :].unsqueeze(0) # (1, gen_prompt_len, Q)
    print(f"Using generation prompt (from style_prompt end) of shape: {generation_prompt_s_indices.shape}")
    print(f"使用生成提示 (来自 style_prompt 末尾)，形状为: {generation_prompt_s_indices.shape}")
    
    generated_sequence = generation_prompt_s_indices.clone() # (B, T_current, Q)，克隆生成提示作为生成序列的开始
    
    current_caches_gen: List[LayerCacheS] = [] # Initialize empty, 初始化空缓存列表
    batch_size_gen = generation_prompt_s_indices.shape[0] # 生成的批次大小 (这里是1)
    # Initialize caches by passing the generation_prompt through the main model's forward pass
    # Note: The style_vector is derived from the longer style_prompt_s_indices.
    # When processing the generation_prompt to fill caches, we use this precomputed_style_vector.
    # 通过将 generation_prompt 传递给主模型的 forward 方法来初始化缓存
    # 注意: style_vector 是从较长的 style_prompt_s_indices 派生的。
    # 在处理 generation_prompt 以填充缓存时，我们使用这个预计算的 style_vector。
    
    # For the forward pass to fill caches, we need to give the style_encoder *something*.
    # It's best if it's the same sequence that style_vector was computed from, or compatible.
    # Here, we pass the full style_prompt_list and style_lengths again for cache init.
    # The s_indices_input for cache init is generation_prompt_s_indices.
    # 对于填充缓存的 forward 传递，我们需要给 style_encoder *一些东西*。
    # 最好是与计算 style_vector 时使用的序列相同或兼容的序列。
    # 这里，我们再次传递完整的 style_prompt_list 和 style_lengths 用于缓存初始化。
    # 用于缓存初始化的 s_indices_input 是 generation_prompt_s_indices.

    print("Processing generation prompt to fill initial caches (with precomputed style vector)...")
    print("处理生成提示以填充初始缓存 (使用预计算的风格向量)...")
    with torch.no_grad():
        # Project style vector once for repeated use in forward/step
        # 投影一次风格向量，以便在 forward/step 中重复使用
        projected_style_for_cache_init = model.style_proj_for_addition(precomputed_style_vector) # (B, d_model)

        # Simplified cache initialization:
        # 简化的缓存初始化:
        # Embed the generation_prompt_s_indices
        # 嵌入 generation_prompt_s_indices
        gen_prompt_embeddings_list = []
        for q_idx in range(model.config.dac_num_quantizers):
            gen_prompt_embeddings_list.append(model.s_embeddings[q_idx](generation_prompt_s_indices[:, :, q_idx]))
        
        x_gen_prompt = torch.cat(gen_prompt_embeddings_list, dim=-1)
        x_gen_prompt = model.s_input_projection(x_gen_prompt) # (B, gen_prompt_len, d_model)
        x_gen_prompt = x_gen_prompt + projected_style_for_cache_init.unsqueeze(1) # Add style

        # Pad x_gen_prompt for cache initialization if Mamba layers are present and require it
        x_for_cache_init_loop = x_gen_prompt
        current_prompt_seq_len = x_gen_prompt.shape[1]
        
        # Determine if any Mamba layer is used in the model
        is_any_mamba_layer = False
        if hasattr(model.config, 'use_attention_layer_indices') and hasattr(model.config, 'n_layer'):
            is_any_mamba_layer = any(not (idx in (model.config.use_attention_layer_indices or [])) for idx in range(model.config.n_layer))

        if is_any_mamba_layer and hasattr(model.config, 'chunk_size') and model.config.chunk_size > 0 and \
           current_prompt_seq_len > 0 and (current_prompt_seq_len % model.config.chunk_size != 0):
            
            target_len_for_cache_init = ((current_prompt_seq_len - 1) // model.config.chunk_size + 1) * model.config.chunk_size
            num_padding = target_len_for_cache_init - current_prompt_seq_len
            # Pad the sequence dimension (dim 1) of x_gen_prompt.
            # Pad tuple is (pad_D_left, pad_D_right, pad_T_left, pad_T_right, pad_B_left, pad_B_right)
            # For x_gen_prompt (B, T, D), we pad T (dim 1) at the end.
            print(f"Padding generation prompt for cache init from {current_prompt_seq_len} to {target_len_for_cache_init}")
            x_for_cache_init_loop = F.pad(x_gen_prompt, (0,0, 0,num_padding, 0,0), value=0.0)

        # Initialize empty caches first
        for _ in range(model.config.n_layer):
            is_attn = _ in (model.config.use_attention_layer_indices or []) # 判断是否为注意力层
            mamba_c_init = MambaInferenceCache.alloc(batch_size_gen, model.config, device=device) if not is_attn else None
            attn_c_init = None
            current_caches_gen.append((mamba_c_init, attn_c_init))

        # Pass through layers to populate caches (attn_mask assumed None for unpadded short prompt)
        # Use x_for_cache_init_loop which might be padded
        for i, layer in enumerate(model.layers):
            x_for_cache_init_loop, current_caches_gen[i] = layer(x_for_cache_init_loop, layer_cache=current_caches_gen[i], attn_mask=None)
            
    print("Initial caches populated for generation.")
    print("已填充用于生成的初始缓存。")

    # 获取生成提示的最后一个时间步作为下一次生成的输入
    last_s_step_for_gen = generation_prompt_s_indices[:, -1, :] # (B, Q)
    max_gen_len = 150 # Generate more tokens, 生成更多标记
    print(f"Starting generation from last prompt token: {last_s_step_for_gen.shape}, Target len: {max_gen_len}")

    all_eos_predicted = False # 标记是否所有量化器都预测了 EOS
    with torch.no_grad(): # 不计算梯度进行生成
        for t_step in range(max_gen_len): # 循环生成每个时间步
            # 模型单步生成
            logits_next_s, current_caches_gen = model.step(
                s_step_indices=last_s_step_for_gen, 
                caches=current_caches_gen,
                style_vector=precomputed_style_vector # Pass the same precomputed style vector
                                                      # 传递相同的预计算风格向量
            )
            # 从 logits 中选择概率最大的标记作为预测结果
            predicted_s_step = torch.argmax(logits_next_s, dim=-1) # (B, 1, Q)
            # 将预测的标记拼接到已生成的序列中
            generated_sequence = torch.cat((generated_sequence, predicted_s_step), dim=1)
            
            all_eos_predicted = True # 先假设所有量化器都预测了 EOS
            for b_idx in range(batch_size_gen): # 遍历批次中的每个样本 (这里只有一个)
                if not torch.all(predicted_s_step[b_idx, 0, :] == EOS_TOKEN): # 如果任何一个量化器没有预测 EOS
                    all_eos_predicted = False; break # 则标记为 False 并跳出循环
            
            if all_eos_predicted: # 如果所有量化器都预测了 EOS
                print(f"EOS predicted by all quantizers at step {t_step+1} after prompt. Total len: {generated_sequence.shape[1]}")
                print(f"在提示后的第 {t_step+1} 步，所有量化器均预测到 EOS。总长度: {generated_sequence.shape[1]}")
                break # 停止生成
            
            last_s_step_for_gen = predicted_s_step.squeeze(1)
            if (t_step + 1) % 20 == 0: # 每生成20步打印一次信息
                print(f"Generated step {t_step+1}, current sequence length: {generated_sequence.shape[1]}")
                print(f"已生成 {t_step+1} 步，当前序列长度: {generated_sequence.shape[1]}")
    
    if not all_eos_predicted : print(f"Max gen length {max_gen_len} reached.") # 如果达到最大生成长度仍未预测 EOS
    print(f"Finished generation. Final sequence shape: {generated_sequence.shape}")
    print(f"生成完成。最终序列形状: {generated_sequence.shape}")
    
    # --- 保存和加载模型 ---
    save_dir_s_styled = "./jamba_s_in_s_out_styled_model_example" # 保存目录
    model.save_pretrained(save_dir_s_styled) # 保存模型
    try:
        # 加载已保存的模型
        loaded_model_s_styled = JambaForAudioGeneration_S_In_S_Out.from_pretrained(save_dir_s_styled, device=device)
        print("S-in-S-out STYLED model loaded successfully.")
        print("S输入S输出 (带风格) 模型加载成功。")
    except Exception as e:
        print(f"Failed to load S-in-S-out STYLED model: {e}")
        print(f"加载 S输入S输出 (带风格) 模型失败: {e}")

    print("S-in-S-out STYLED example finished.") 
    print("S输入S输出 (带风格) 示例完成。") 