import os
from dataclasses import asdict

import torch
import torch.nn as nn
from torch._C._nn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import re
import shutil
import datetime
import torch.nn.functional as F
import torch_optimizer
from torch.amp import GradScaler, autocast

# Imports for the new model
from jamba_audio_s_in_s_out import JambaSConfig, JambaForAudioGeneration_S_In_S_Out, jamba_s_styled_collate_fn, MambaInferenceCache # 关键导入，添加 MambaInferenceCache
from functools import partial # 用于 collate_fn

# dacFunction 不再需要用于 s_to_zq，但 load_encoded_tensors 可能会从中调整
# from dac44100hz import dacFunction
# get_model 被直接实例化模型或 from_pretrained 取代
# from get_model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
# 此处不再需要来自 s_to_zq_differentiable_custom 的 EPS_FOR_NORM。
# 如果 JambaSConfig 或其组件需要它，则应在那里定义。

TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'

# --- 配置 ---
# 通用路径和设置
# DAC_PT_DIR: 包含带有 's' (以及可选的 'zq') 张量的 .pt 文件的目录
# ubuntu 配置
DAC_PT_DIR = '/root/autodl-tmp/modelTrain/jambaDataset2'
# DAC_CODEBOOKS_FILE_PATH 用于 s_to_zq，在此脚本级别不再需要。

MAX_RECENT_CHECKPOINTS_TO_KEEP = 5
BEST_MODEL_SUBDIR = "best_model_s_in_s_out" # 此模型的新子目录

VALIDATION_INTERVAL_FILES = 50
VALIDATION_NUM_FILES = 10

# PRETRAINED_MODEL_LOAD_PATH: 加载预训练的 S-in-S-out 模型的路径 (例如，来自先前运行)
# 示例: './jamba_s_in_s_out_model_trained/best_model_s_in_s_out'
PRETRAINED_MODEL_LOAD_PATH: Optional[str] = ""

# STEP_MODEL_SAVE_DIR: 保存此脚本训练的模型的目录
STEP_MODEL_SAVE_DIR = './jamba_s_in_s_out_model_step_trained'
LOG_FILE_PATH = os.path.join(STEP_MODEL_SAVE_DIR, "training_log_s_in_s_out_step.log")

# 新增：控制是否冻结StyleEncoder的参数
FREEZE_STYLE_ENCODER_PARAMS = False

# 训练参数 (许多可以重用)
TOTAL_TRAINING_EPOCHS = 5 # 总训练周期数
EPOCHS_FILES_NUM = 100 # 每个epoch从训练集处理的文件数
LEARNING_RATE = 2e-4 # 学习率
# FILE_LOAD_BATCH_SIZE: 从磁盘加载并按顺序处理的 .pt 文件数量。
# 这与模型处理批次大小不同。
FILE_PROCESSING_BATCH_SIZE = 2 # 一次处理一个完整文件的数据，然后再处理下一个 .pt 文件。
                               # 块的内部批处理将由 MODEL_STEP_PROCESSING_BATCH_SIZE 处理

MODEL_PROCESSING_BATCH_SIZE_STEP = 5 # model.step() 并行处理的序列 (块) 数量 (原 MODEL_STEP_PROCESSING_BATCH_SIZE)
CHUNK_MAX_TIME_STEPS_STEP = 256 # 单个数据块的最大时间步长 (序列长度) (原 CHUNK_MAX_TIME_STEPS)

FORWARD_MODEL_PROCESSING_BATCH_SIZE = 5 # FORWARD 路径并行处理的序列 (块) 数量
FORWARD_CHUNK_MAX_TIME_STEPS = 4096    # FORWARD 路径单个数据块的最大时间步长

TRAIN_EOS_ONLY_AT_VERY_END = True # 是否仅在整个序列的非常末尾添加EOS进行训练 (影响目标构建)
CLIP_GRAD_NORM = 1.0 # 梯度裁剪的范数阈值，设为0或负数则不进行梯度裁剪
# ALWAYS_USE_PREDICTED_ZQ_AFTER_FIRST_STEP 已移除 (ZQ 不是主要输入)

FORWARD_TRAINING_PROBABILITY = 0.6 # 使用 model.forward() 进行训练的概率 (0.0 到 1.0)

GRAD_ACCUMULATION_STEPS = 4 # 梯度累积的步数，用于模拟更大的批次大小
USE_MIXED_PRECISION = True # 是否在CUDA上使用混合精度训练 (自动使用float16)

SCHEDULED_SAMPLING_INITIAL_EPSILON = 1.0 # 计划采样中，使用真实标签 (teacher forcing) 的初始概率
SCHEDULED_SAMPLING_FINAL_EPSILON = 0.05  # 计划采样中，使用真实标签的最终概率 (会线性衰减到此值)
SCHEDULED_SAMPLING_DECAY_STEPS = 10000  # 计划采样epsilon从初始值衰减到最终值的步数 (文件处理次数)
# AUX_ZQ_LOSS_WEIGHT 已移除。
# GUMBEL_SOFTMAX_TAU 设置已移除 (用于 s_to_zq)。
# 如果将 Gumbel 用于采样 S，则可能需要新的参数，但从 argmax/multinomial 开始。

NUM_PASSES_PER_FILE_IN_EPOCH = 2 # 在每个训练周期 (epoch) 中，训练集中的每个文件被完整处理的次数

TEACHER_FORCING_EPOCH_INTERVAL = 10 # 每处理多少个文件后，执行一次专门的教师强制周期
TEACHER_FORCING_EPOCH_NUM_FILES = 2  # 在专门的教师强制周期中，使用多少个文件进行训练
TEACHER_FORCING_EPOCH_LEARNING_RATE = 1e-4 # 专门的教师强制周期的学习率
TEACHER_FORCING_EPOCH_CLIP_GRAD_NORM = 1.0 # 专门的教师强制周期的梯度裁剪范数
# TEACHER_FORCING_EPOCH_AUX_ZQ_LOSS_WEIGHT 已移除。
TEACHER_FORCING_EPOCH_EFFECTIVE_EPSILON = 1.0 # S 的完全教师强制，在教师强制周期中epsilon固定为1.0
# TEACHER_FORCING_EPOCH_FIXED_TAU 已移除。

# --- JambaSConfig 特定参数 (需要根据所需的模型架构进行定义) ---
# 这些应与您希望如何为模型定义 JambaSConfig 相一致。
# 此处的值是示例，应进行检查。
JAMBA_S_CONFIG_PARAMS = {
    "d_model": 768,
    "n_layer": 8, # 示例，请检查
    "d_state": 64, # Mamba 特有
    "d_conv": 4,   # Mamba 特有
    "expand": 2,   # Mamba 特有
    "headdim": 64, # Mamba2 特有，或用于注意力机制的通用参数
    "chunk_size": 128, # 与数据分块关联，用于潜在的 Mamba 优化

    "num_attention_heads": 8,
    "attention_dropout": 0.1,
    "attention_window_size": 256, # 注意力滑动窗口大小，None或0表示不使用
    # 示例: 从第1层开始，每隔一层使用注意力机制
    "use_attention_layer_indices": [i for i in range(8) if i % 2 != 0], # 如果更改了 n_layer，请调整

    "dac_codebook_size": 1025, # 重要：如果使用 EOS，请将其设置为您的实际码本大小 + 1
    "dac_num_quantizers": 9,   # 重要：请将其设置为 DAC 的实际量化器数量

    "s_embedding_dim_individual": 64, # 主模型的 S 输入的嵌入维度

    "style_dim": 64,
    "style_s_embedding_dim_individual": 64,
    "style_encoder_num_conv_layers": 4,
    "style_encoder_kernel_size": 5,
}
# EOS 标记 ID，如果 EOS 是最后一个标记，则应为 dac_codebook_size - 1
EOS_TOKEN_ID = JAMBA_S_CONFIG_PARAMS["dac_codebook_size"] - 1
PAD_TOKEN_S_VALUE = -100 # 用于损失 ignore_index 的目标填充值

# --- 检查点管理 (可从 train_autoregressive_step.py 重用) ---
def parse_checkpoint_name(dir_name: str) -> Optional[int]:
    match = re.fullmatch(r"checkpoint_file_(\d+)_.*", dir_name)
    if match:
        return int(match.group(1))
    return None

def find_latest_checkpoint_path_and_counter(base_dir: str, _log_func) -> Tuple[Optional[str], int]:
    if not os.path.isdir(base_dir):
        return None, 0
    checkpoints = []
    try:
        for item_name in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item_name)
            if os.path.isdir(item_path):
                counter = parse_checkpoint_name(item_name)
                if counter is not None:
                    checkpoints.append({'path': item_path, 'counter': counter, 'name': item_name})
    except OSError as e:
        _log_func(f"警告：查找检查点时访问目录 '{base_dir}' 出错: {e}")
        return None, 0
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=lambda x: x['counter'], reverse=True)
    latest = checkpoints[0]
    return latest['path'], latest['counter']

def manage_checkpoints(base_dir: str, max_to_keep: int, _log_func):
    if not os.path.isdir(base_dir) or max_to_keep <= 0:
        return
    all_checkpoints = []
    try:
        for item_name in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item_name)
            if os.path.isdir(item_path):
                counter = parse_checkpoint_name(item_name)
                if counter is not None:
                    all_checkpoints.append({'path': item_path, 'counter': counter, 'name': item_name})
    except OSError as e:
        _log_func(f"警告：管理检查点时访问目录 '{base_dir}' 出错: {e}")
        return
    if len(all_checkpoints) <= max_to_keep:
        return
    all_checkpoints.sort(key=lambda x: x['counter'])
    num_to_delete = len(all_checkpoints) - max_to_keep
    checkpoints_to_delete = all_checkpoints[:num_to_delete]
    _log_func(f"  将保留 {max_to_keep} 个检查点，尝试删除 {len(checkpoints_to_delete)} 个旧检查点...")
    for cp_info in checkpoints_to_delete:
        try:
            shutil.rmtree(cp_info['path'])
            _log_func(f"    已删除旧检查点: {cp_info['path']}")
        except OSError as e:
            _log_func(f"    删除旧检查点 {cp_info['path']} 失败: {e}")

# --- Data Loading Helpers (adapted for S focus) ---
def get_all_pt_files(directory: str) -> List[str]:
    pt_files: List[str] = []
    if not os.path.isdir(directory):
        print(f"错误：'{directory}' 不是一个有效的目录。")
        return pt_files
    try:
        for entry_name in os.listdir(directory):
            file_path = os.path.join(directory, entry_name)
            if os.path.isfile(file_path) and entry_name.endswith(".pt"):
                pt_files.append(file_path)
        pt_files.sort()
    except OSError as e:
        print(f"错误：访问目录 '{directory}' 时发生错误: {e}")
    return pt_files

def load_s_tensor_from_file(load_path: str, target_device: torch.device) -> Optional[torch.Tensor]:
    """
    从 .pt 文件加载 's' 张量。
    假设 .pt 文件包含一个至少带有 's' 键的字典。
    's' 张量期望来自 DAC 的形状为 (1, Q, T_orig) 或 (Q, T_orig)，需要转换为 (T_orig, Q)。
    """
    try:
        if os.path.exists(load_path):
            loaded_data = torch.load(load_path, map_location=CPU_DEVICE, weights_only=True)
            if 's' not in loaded_data:
                print(f"错误：在 {load_path} 中未找到 's' 键")
                return None
            
            s_loaded = loaded_data['s'].to(target_device) # 加载到CPU后移动到目标设备

            # 重塑 s：模型和风格编码器期望的形状为 (T_orig, Q)
            if s_loaded.ndim == 3 and s_loaded.shape[0] == 1: # (1, Q, T_orig)
                s_loaded = s_loaded.squeeze(0).permute(1, 0)  # -> (Q, T_orig) -> (T_orig, Q)
            elif s_loaded.ndim == 2: # 假设为 (Q, T_orig)
                s_loaded = s_loaded.permute(1, 0) # -> (T_orig, Q)
            elif s_loaded.ndim == 2 and s_loaded.shape[1] == JAMBA_S_CONFIG_PARAMS["dac_num_quantizers"]: # 已经是 (T_orig, Q)
                pass # 正确的形状
            else:
                print(f"错误：{load_path} 中的 's' 张量形状异常: {s_loaded.shape}。预期形状为 (1,Q,T) 或 (Q,T) 或 (T,Q)。")
                return None
            
            # Validate Q dimension
            if s_loaded.shape[1] != JAMBA_S_CONFIG_PARAMS["dac_num_quantizers"]:
                print(f"错误：加载的 's' 张量 Q 维度 ({s_loaded.shape[1]}) 与配置 ({JAMBA_S_CONFIG_PARAMS['dac_num_quantizers']}) 在 {load_path} 中不匹配")
                return None

            # --- 新增：验证 S 码元索引值的范围 ---
            min_val = torch.min(s_loaded)
            max_val = torch.max(s_loaded)
            codebook_size = JAMBA_S_CONFIG_PARAMS["dac_codebook_size"]
            if not (min_val >= 0 and max_val < codebook_size):
                print(f"错误：文件 {load_path} 中的 's' 张量包含超出范围 [0, {codebook_size-1}] 的标记ID。检测到范围: [{min_val.item()}, {max_val.item()}]。跳过此文件。")
                return None
            # --- 验证结束 ---

            return s_loaded
        else:
            print(f"错误：未找到已保存的编码文件 {load_path}")
            return None
    except Exception as e:
        print(f"错误：从 '{load_path}' 加载 's' 张量时出错: {e}")
        return None

# This function might be less relevant if we use a proper DataLoader with collate_fn
# but keeping a similar structure for now.
def custom_pad_s_sequences(
    sequences_on_cpu: List[torch.Tensor],  # (T_orig, Q) 的列表
    target_device: torch.device,
    padding_value: int, # 例如，PAD_TOKEN_S_VALUE 或 EOS_TOKEN_ID
    batch_first: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    填充 S 序列列表 (T_orig, Q) 并返回堆叠的张量和长度。
    移动到目标设备。
    """
    if not sequences_on_cpu:
        # 如果下游需要，则返回适当形状的空张量，或引发错误
        num_q = JAMBA_S_CONFIG_PARAMS["dac_num_quantizers"]
        return torch.empty((0,0,num_q), device=target_device), torch.empty((0), dtype=torch.long, device=target_device)

    # 首先移动到目标设备
    sequences_on_target_device = [s.to(target_device) for s in sequences_on_cpu]

    # pad_sequence 期望 (T, *) 或 (B, T, *) 如果 batch_first=True
    # 我们的序列是 (T_orig, Q)。pad_sequence 将填充 T_orig。
    # 如果 batch_first=True，输出是 (B, T_max, Q)
    # 如果 batch_first=False，输出是 (T_max, B, Q)
    
    lengths = torch.tensor([s.shape[0] for s in sequences_on_target_device], dtype=torch.long, device=target_device)
    
    padded_sequences = pad_sequence(
        sequences_on_target_device,
        batch_first=batch_first,
        padding_value=float(padding_value) # pad_sequence 期望 padding_value 为浮点数
    )
    return padded_sequences, lengths


# --- Main Training Function ---
def train_model_step_by_step():
    os.makedirs(STEP_MODEL_SAVE_DIR, exist_ok=True)
    
    log_f = open(LOG_FILE_PATH, 'a', encoding='utf-8') # 指定UTF-8编码
    def _log(msg):
        print(msg) # 控制台输出
        log_f.write(f"{msg}\n") # 写入日志文件
        log_f.flush()

    _log(f"\n--- 新的 S输入S输出 训练会话开始于: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    _log(f"使用 JambaForAudioGeneration_S_In_S_Out 模型。")
    _log(f"主要计算设备: {DEVICE}")
    _log(f"数据加载和初始处理设备: {CPU_DEVICE}")

    # --- 模型配置和初始化 ---
    config = JambaSConfig(**JAMBA_S_CONFIG_PARAMS)
    _log(f"JambaSConfig 初始化完成: {config}")
    
    model: Optional[JambaForAudioGeneration_S_In_S_Out] = None
    global_files_processed_counter = 0

    _log("--- 开始加载模型 --- ")
    # 阶段 1: 尝试从此脚本特定的最新文件检查点加载
    latest_ckpt_path, ckpt_counter = find_latest_checkpoint_path_and_counter(STEP_MODEL_SAVE_DIR, _log)
    if latest_ckpt_path:
        _log(f"  找到最新的逐步训练检查点: {latest_ckpt_path} (文件计数器 {ckpt_counter})")
        try:
            model = JambaForAudioGeneration_S_In_S_Out.from_pretrained(latest_ckpt_path, device=DEVICE)
            config = model.config # 使用加载模型的配置
            global_files_processed_counter = ckpt_counter
            _log(f"  成功从逐步训练的检查点加载模型。全局文件处理计数器已恢复为 {global_files_processed_counter}。")
        except Exception as e:
            _log(f"  从逐步训练的检查点 {latest_ckpt_path} 加载模型失败: {e}")
            model = None

    # 阶段 2: 如果没有步骤检查点，则尝试 PRETRAINED_MODEL_LOAD_PATH
    if model is None and PRETRAINED_MODEL_LOAD_PATH and os.path.isdir(PRETRAINED_MODEL_LOAD_PATH):
        _log(f"阶段2：尝试从 PRETRAINED_MODEL_LOAD_PATH ('{PRETRAINED_MODEL_LOAD_PATH}') 加载...")
        try:
            model = JambaForAudioGeneration_S_In_S_Out.from_pretrained(PRETRAINED_MODEL_LOAD_PATH, device=DEVICE)
            config = model.config # 使用加载模型的配置
            _log(f"  成功从 {PRETRAINED_MODEL_LOAD_PATH} 加载预训练模型。")
        except Exception as e:
            _log(f"  从 {PRETRAINED_MODEL_LOAD_PATH} 加载预训练模型失败: {e}")
            model = None
    elif model is None and PRETRAINED_MODEL_LOAD_PATH:
         _log(f"阶段2：PRETRAINED_MODEL_LOAD_PATH ('{PRETRAINED_MODEL_LOAD_PATH}') 不是一个有效目录，跳过。")

    # 阶段 3: 如果仍然没有模型，则尝试从此脚本的保存目录加载 "best_model"
    if model is None:
        best_model_dir = os.path.join(STEP_MODEL_SAVE_DIR, BEST_MODEL_SUBDIR)
        if os.path.isdir(best_model_dir):
            _log(f"阶段3：尝试从最佳模型目录 ('{best_model_dir}') 加载...")
            try:
                model = JambaForAudioGeneration_S_In_S_Out.from_pretrained(best_model_dir, device=DEVICE)
                config = model.config
                _log(f"  成功从 {best_model_dir} 加载最佳模型。")
            except Exception as e:
                _log(f"  从 {best_model_dir} 加载最佳模型失败: {e}")
                model = None
        else:
            _log(f"阶段3：未找到最佳模型目录 ('{best_model_dir}')。")
    
    # 阶段 4: 如果仍然没有模型，则使用当前的 JambaSConfig 初始化一个新模型
    if model is None:
        _log(f"阶段4：所有加载尝试均失败。正在初始化一个新的 JambaForAudioGeneration_S_In_S_Out 模型。")
        global_files_processed_counter = 0
        try:
            current_config_for_new_model = JambaSConfig(**JAMBA_S_CONFIG_PARAMS) # 使用定义的参数
            model = JambaForAudioGeneration_S_In_S_Out(current_config_for_new_model, device=DEVICE)
            config = model.config # 这是 current_config_for_new_model
            _log(f"  新模型已使用配置初始化: {config}")
        except Exception as e:
            _log(f"  致命错误：无法初始化新模型: {e}。正在退出。")
            log_f.close()
            return
            
    if model is None: # 如果新模型初始化不成功，则不应发生
        _log(f"致命错误：模型未能成功加载或初始化。正在退出。")
        log_f.close()
        return

    _log("--- 模型加载/初始化完成 --- ")
    _log(f"当前活动模型配置: {asdict(config)}") # 记录活动配置
    _log(f"全局文件处理计数器初始值: {global_files_processed_counter}")
    _log(f"计划采样: 初始epsilon={SCHEDULED_SAMPLING_INITIAL_EPSILON}, 最终epsilon={SCHEDULED_SAMPLING_FINAL_EPSILON}, 衰减步数={SCHEDULED_SAMPLING_DECAY_STEPS}")
    _log(f"每个Epoch内，单个文件的训练遍数: {NUM_PASSES_PER_FILE_IN_EPOCH}")
    _log(f"验证: 文件间隔={VALIDATION_INTERVAL_FILES}, 文件数量={VALIDATION_NUM_FILES}")
    _log(f"梯度累积步数: {GRAD_ACCUMULATION_STEPS}")
    mixed_precision_status = '启用' if USE_MIXED_PRECISION and DEVICE.type == 'cuda' else \
                             ('禁用 (CUDA不可用)' if USE_MIXED_PRECISION else '禁用 (配置未开启)')
    _log(f"混合精度训练: {mixed_precision_status}")
    _log(f"用于输入填充和目标构建的EOS标记ID: {EOS_TOKEN_ID}")
    _log(f"用于损失ignore_index的PAD标记值: {PAD_TOKEN_S_VALUE}")


    model.to(DEVICE)
    # torch.compile(model) # 可选：在验证正确性后考虑

    all_files_from_source = get_all_pt_files(DAC_PT_DIR)
    if not all_files_from_source:
        _log("未找到 .pt 文件。正在退出。")
        log_f.close()
        return
    random.shuffle(all_files_from_source)

    validation_files: List[str] = []
    train_files_full_list_initial: List[str] = []
    if VALIDATION_NUM_FILES > 0 and len(all_files_from_source) > VALIDATION_NUM_FILES:
        validation_files = all_files_from_source[:VALIDATION_NUM_FILES]
        train_files_full_list_initial = all_files_from_source[VALIDATION_NUM_FILES:]
        _log(f"已划分验证集: {len(validation_files)} 个文件。训练文件池: {len(train_files_full_list_initial)} 个文件。")
    else:
        _log(f"警告：文件总数不足以进行验证，或已禁用验证。所有文件将用于训练。")
        train_files_full_list_initial = all_files_from_source
    
    if not train_files_full_list_initial:
        _log("错误：划分后训练文件列表为空。请检查 EPOCHS_FILES_NUM 和 VALIDATION_NUM_FILES 的设置。正在退出。")
        log_f.close()
        return

    num_train_files_to_use_for_epoch = min(len(train_files_full_list_initial), EPOCHS_FILES_NUM)

    optimizer = torch_optimizer.RAdam(model.parameters(), lr=LEARNING_RATE)
    for group in optimizer.param_groups: # 为恢复调度器状态所需
        group['initial_lr'] = group['lr']
    
    total_steps_for_scheduler = SCHEDULED_SAMPLING_DECAY_STEPS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps_for_scheduler, eta_min=0, last_epoch=global_files_processed_counter -1)
    _log(f"学习率调度器: CosineAnnealingLR, T_max={total_steps_for_scheduler} (SCHEDULED_SAMPLING_DECAY_STEPS), last_epoch={scheduler.last_epoch}")
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_S_VALUE) # PAD_TOKEN_S_VALUE 为 -100

    if FREEZE_STYLE_ENCODER_PARAMS:
        if hasattr(model, 'style_encoder') and model.style_encoder is not None:
            _log("尝试冻结 StyleEncoder 参数...")
            for param in model.style_encoder.parameters():
                param.requires_grad = False
            if all(not p.requires_grad for p in model.style_encoder.parameters()):
                _log("StyleEncoder 参数已成功冻结。")
            else:
                _log("警告: StyleEncoder 参数冻结失败或部分失败。")
        else:
            _log("未找到 StyleEncoder 或其为 None，跳过冻结 (根据配置)。")
    else:
        _log("StyleEncoder 参数将不会被冻结 (根据配置)，将参与训练。")
        if hasattr(model, 'style_encoder') and model.style_encoder is not None:
            for param in model.style_encoder.parameters():
                param.requires_grad = True # 确保它们是可训练的
            if all(p.requires_grad for p in model.style_encoder.parameters()):
                _log("StyleEncoder 参数已设置为可训练。")
            else:
                _log("警告: StyleEncoder 某些参数未能成功设置为可训练。")
        else:
            _log("未找到 StyleEncoder 或其为 None，无需操作。")


    scaler = torch.amp.GradScaler(enabled=(USE_MIXED_PRECISION and DEVICE.type == 'cuda'))
    best_avg_val_loss = float('inf')

    for epoch in tqdm(range(TOTAL_TRAINING_EPOCHS), desc="训练周期 (Epochs)", unit="周期"):
        _log(f"\n--- 开始常规训练周期 {epoch + 1}/{TOTAL_TRAINING_EPOCHS} ---")
        model.train()
        epoch_total_main_loss = 0.0
        num_loss_contributions_in_epoch = 0

        current_epoch_train_files = random.sample(train_files_full_list_initial, k=num_train_files_to_use_for_epoch)
        _log(f"  周期 {epoch + 1}: 已选择 {len(current_epoch_train_files)} 个文件用于训练。")

        for file_idx, file_path in enumerate(tqdm(current_epoch_train_files, desc=f"常规周期 {epoch+1} 文件", unit="文件")):
            for pass_num_in_file in range(NUM_PASSES_PER_FILE_IN_EPOCH):
                _log(f"  文件 '{os.path.basename(file_path)}', 第 {pass_num_in_file + 1}/{NUM_PASSES_PER_FILE_IN_EPOCH} 遍训练开始。")

                s_full_single_cpu: Optional[torch.Tensor] = load_s_tensor_from_file(file_path, CPU_DEVICE)
                if s_full_single_cpu is None:
                    _log(f"    警告：从 {file_path} 加载 's' 张量失败，跳过。")
                    continue
                
                s_originals_for_style_list = [s_full_single_cpu.to(model.get_style_encoder().device)]
                lengths_for_style_tensor = torch.tensor([s_full_single_cpu.shape[0]], dtype=torch.long, device=model.get_style_encoder().device)
                
                style_vector_this_file_gpu: Optional[torch.Tensor] = None
                try:
                    with torch.no_grad():
                        style_vector_this_file_gpu = model.get_style_encoder()(s_originals_for_style_list, lengths_for_style_tensor)
                    style_vector_this_file_gpu = style_vector_this_file_gpu.to(DEVICE)
                except Exception as e_style:
                    _log(f"    为 {file_path} 计算风格向量时出错: {e_style}。跳过此文件遍次。")
                    continue

                if style_vector_this_file_gpu is None:
                    _log(f"    计算后 {file_path} 的风格向量为 None。跳过此文件遍次。")
                    continue
                
                s_input_seq_full = s_full_single_cpu[:-1, :]
                s_target_seq_full = s_full_single_cpu[1:, :]
                eos_row_for_s = torch.full((1, config.dac_num_quantizers), EOS_TOKEN_ID, dtype=s_target_seq_full.dtype, device=CPU_DEVICE)
                s_target_seq_full_with_eos = torch.cat((s_target_seq_full, eos_row_for_s), dim=0)
                
                current_file_pass_main_loss = 0.0
                num_loss_contributions_this_file_pass = 0

                current_path_is_forward = random.random() < FORWARD_TRAINING_PROBABILITY
                # 初始化将在分支中更新的日志变量
                # current_batch_loss_for_log = 0.0  # These are now updated inside the j_idx loop before logging
                # current_epsilon_for_log = -1.0   
                # actual_batch_size_for_log_trigger = 0 

                if current_path_is_forward:
                    _log(f"  文件 '{os.path.basename(file_path)}', 第 {pass_num_in_file + 1} 遍: 使用 FORWARD 路径训练 (Chunk: {FORWARD_CHUNK_MAX_TIME_STEPS}, Batch: {FORWARD_MODEL_PROCESSING_BATCH_SIZE})")
                    
                    forward_s_input_chunks_cpu = []
                    forward_s_target_chunks_cpu = []
                    total_time_steps_for_chunks_fw = s_input_seq_full.shape[0]

                    for chunk_start_idx in range(0, total_time_steps_for_chunks_fw, FORWARD_CHUNK_MAX_TIME_STEPS):
                        chunk_end_idx = min(chunk_start_idx + FORWARD_CHUNK_MAX_TIME_STEPS, total_time_steps_for_chunks_fw)
                        s_input_chunk_cpu = s_input_seq_full[chunk_start_idx:chunk_end_idx, :]
                        s_target_chunk_cpu = s_target_seq_full_with_eos[chunk_start_idx:chunk_end_idx, :]
                        if s_input_chunk_cpu.shape[0] == 0: continue
                        forward_s_input_chunks_cpu.append(s_input_chunk_cpu)
                        forward_s_target_chunks_cpu.append(s_target_chunk_cpu)
                    
                    if not forward_s_input_chunks_cpu:
                        _log(f"    FORWARD路径: 文件 {file_path} 没有可处理的数据块。跳过此文件遍次内当前路径的训练。")
                    else:
                        num_chunk_batches_fw = (len(forward_s_input_chunks_cpu) + FORWARD_MODEL_PROCESSING_BATCH_SIZE - 1) // FORWARD_MODEL_PROCESSING_BATCH_SIZE
                        base_desc_fw = f"FWD F{file_idx+1}/{len(current_epoch_train_files)} P{pass_num_in_file+1}" # 基础描述
                        
                        chunk_pbar_fw = tqdm(range(num_chunk_batches_fw), desc=base_desc_fw, unit="batch", leave=False, dynamic_ncols=True)
                        for j_idx in chunk_pbar_fw:
                            j_start_offset = j_idx * FORWARD_MODEL_PROCESSING_BATCH_SIZE
                            current_s_input_chunks_cpu_batch = forward_s_input_chunks_cpu[j_start_offset : j_start_offset + FORWARD_MODEL_PROCESSING_BATCH_SIZE]
                            current_s_target_chunks_cpu_batch = forward_s_target_chunks_cpu[j_start_offset : j_start_offset + FORWARD_MODEL_PROCESSING_BATCH_SIZE]

                            if not current_s_input_chunks_cpu_batch: continue
                            
                            s_input_batch_padded_fw, original_lengths_input_fw = custom_pad_s_sequences(
                                current_s_input_chunks_cpu_batch, DEVICE, padding_value=EOS_TOKEN_ID, batch_first=True
                            )
                            s_target_batch_padded_fw, _ = custom_pad_s_sequences(
                                current_s_target_chunks_cpu_batch, DEVICE, padding_value=PAD_TOKEN_S_VALUE, batch_first=True
                            )
                            
                            actual_step_batch_size_fw = s_input_batch_padded_fw.shape[0]
                            max_len_this_step_batch_fw = s_input_batch_padded_fw.shape[1]

                            arange_fw = torch.arange(max_len_this_step_batch_fw, device=DEVICE)[None, :]
                            attn_mask_fw = (arange_fw >= original_lengths_input_fw[:, None]).unsqueeze(1).unsqueeze(1)

                            s_full_single_cpu_on_style_device = s_full_single_cpu.to(model.get_style_encoder().device)
                            
                            # --- 修改开始 ---
                            # 限制送入 StyleEncoder 的序列长度
                            # 使用 FORWARD_CHUNK_MAX_TIME_STEPS (用户设置为12288) 作为风格提取长度的实际 上限
                            # 避免在 StyleEncoder 的初始嵌入拼接时处理过长的完整文件
                            current_total_file_len = s_full_single_cpu_on_style_device.shape[0]
                            # 使用 FORWARD_CHUNK_MAX_TIME_STEPS 作为截断长度，因为它代表了用户期望在FORWARD路径中处理的较长片段
                            max_len_for_style_extraction = FORWARD_CHUNK_MAX_TIME_STEPS 

                            if current_total_file_len > max_len_for_style_extraction:
                                _log(f"    FORWARD StyleEncoder: 将用于风格提取的输入从 {current_total_file_len} 时间步截断为 {max_len_for_style_extraction} 时间步。")
                                s_tensor_for_style_extraction = s_full_single_cpu_on_style_device[:max_len_for_style_extraction, :]
                                actual_len_for_style_calc = max_len_for_style_extraction
                            else:
                                s_tensor_for_style_extraction = s_full_single_cpu_on_style_device
                                actual_len_for_style_calc = current_total_file_len
                            
                            s_originals_for_style_fw = [s_tensor_for_style_extraction]
                            lengths_for_style_fw = torch.tensor([actual_len_for_style_calc], dtype=torch.long, device=model.get_style_encoder().device)
                            # --- 修改结束 ---

                            with torch.amp.autocast(device_type=DEVICE.type, enabled=(USE_MIXED_PRECISION and DEVICE.type == 'cuda')):
                                logits_fw, _ = model.forward(
                                    s_indices_input=s_input_batch_padded_fw,
                                    s_originals_for_style=s_originals_for_style_fw,
                                    lengths_for_style=lengths_for_style_fw,
                                    attn_mask=attn_mask_fw
                                )
                                logits_for_loss_fw = logits_fw.reshape(-1, config.dac_codebook_size)
                                target_for_loss_fw = s_target_batch_padded_fw.reshape(-1)
                                main_loss_fw = loss_fn(logits_for_loss_fw, target_for_loss_fw)

                            current_main_loss_item_fw = main_loss_fw.item()
                            if torch.isnan(main_loss_fw) or torch.isinf(main_loss_fw):
                                # _log(f"    FORWARD警告：检测到 NaN 或 Inf 主损失 ({current_main_loss_item_fw:.4f})。跳过反向传播。文件 {file_idx+1}，块批次 {j_idx+1}。")
                                nan_inf_msg_fw = f"    FORWARD警告：NaN/Inf损失 ({current_main_loss_item_fw:.4f}) @ F{file_idx+1} P{pass_num_in_file+1} B{j_idx+1}. 跳过反向传播."
                                tqdm.write(nan_inf_msg_fw)
                                log_f.write(f"{nan_inf_msg_fw}\n"); log_f.flush()
                                loss_for_backward_scaled = torch.tensor(0.0, device=DEVICE)
                            else:
                                loss_for_backward_scaled = main_loss_fw / GRAD_ACCUMULATION_STEPS
                            
                            scaler.scale(loss_for_backward_scaled).backward()
                            
                            epoch_total_main_loss += current_main_loss_item_fw * actual_step_batch_size_fw
                            current_file_pass_main_loss += current_main_loss_item_fw * actual_step_batch_size_fw
                            num_loss_contributions_in_epoch += actual_step_batch_size_fw
                            num_loss_contributions_this_file_pass += actual_step_batch_size_fw
                            
                            # Optimizer step logic
                            is_last_chunk_batch_for_file_pass = (j_idx == num_chunk_batches_fw - 1)
                            if ((j_idx + 1) % GRAD_ACCUMULATION_STEPS == 0) or is_last_chunk_batch_for_file_pass:
                                if CLIP_GRAD_NORM > 0:
                                    scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                                scaler.step(optimizer)
                                scaler.update()
                                optimizer.zero_grad()
                            
                            # Logging for FORWARD path batch
                            current_lr_fw = optimizer.param_groups[0]['lr']
                            loss_str_fw = f'{current_main_loss_item_fw:.4f}' if not (torch.isnan(torch.tensor(current_main_loss_item_fw)) or torch.isinf(torch.tensor(current_main_loss_item_fw))) else 'NaN/Inf'
                            chunk_pbar_fw.desc = f"{base_desc_fw} | Loss:{loss_str_fw} LR:{current_lr_fw:.1e}"
                # This block replaces the existing "else: # STEP Path" up to just before
# the line: "avg_loss_this_file_pass = current_file_pass_main_loss / num_loss_contributions_this_file_pass..."

                else: # STEP Path
                    progress_ratio_eps_for_step_log = min(1.0, global_files_processed_counter / SCHEDULED_SAMPLING_DECAY_STEPS if SCHEDULED_SAMPLING_DECAY_STEPS > 0 else 1.0)
                    epsilon_for_this_step_path = SCHEDULED_SAMPLING_INITIAL_EPSILON - (SCHEDULED_SAMPLING_INITIAL_EPSILON - SCHEDULED_SAMPLING_FINAL_EPSILON) * progress_ratio_eps_for_step_log
                    _log(f"  文件 '{os.path.basename(file_path)}', 第 {pass_num_in_file + 1} 遍: 使用 STEP 路径训练 (Chunk: {CHUNK_MAX_TIME_STEPS_STEP}, Batch: {MODEL_PROCESSING_BATCH_SIZE_STEP}, Plan Eps: {epsilon_for_this_step_path:.4f})")

                    step_s_input_chunks_cpu: List[torch.Tensor] = []
                    step_s_target_chunks_cpu: List[torch.Tensor] = []
                    # This loop populates step_s_input_chunks_cpu and step_s_target_chunks_cpu
                    # s_input_seq_full and s_target_seq_full_with_eos are defined earlier for the file
                    for chunk_start_idx in range(0, s_input_seq_full.shape[0], CHUNK_MAX_TIME_STEPS_STEP):
                        chunk_end_idx = min(s_input_seq_full.shape[0], chunk_start_idx + CHUNK_MAX_TIME_STEPS_STEP)
                        s_input_chunk_cpu = s_input_seq_full[chunk_start_idx:chunk_end_idx, :]
                        s_target_chunk_cpu = s_target_seq_full_with_eos[chunk_start_idx:chunk_end_idx, :]
                        if s_input_chunk_cpu.shape[0] == 0: continue
                        
                        # Ensure chunks have the full time dimension CHUNK_MAX_TIME_STEPS_STEP by padding if necessary
                        if s_input_chunk_cpu.shape[0] < CHUNK_MAX_TIME_STEPS_STEP:
                            pad_len = CHUNK_MAX_TIME_STEPS_STEP - s_input_chunk_cpu.shape[0]
                            # Pad the time dimension (dim 0 in (T, Q) tensor, so index 2 in F.pad (0,0 for last dim, 0,pad_len for second to last))
                            # Use EOS_TOKEN_ID for input padding, as PAD_TOKEN_S_VALUE (-100) is invalid for embedding lookup
                            s_input_chunk_cpu = F.pad(s_input_chunk_cpu, (0, 0, 0, pad_len), mode='constant', value=EOS_TOKEN_ID)
                            s_target_chunk_cpu = F.pad(s_target_chunk_cpu, (0, 0, 0, pad_len), mode='constant', value=PAD_TOKEN_S_VALUE)
                        
                        step_s_input_chunks_cpu.append(s_input_chunk_cpu)
                        step_s_target_chunks_cpu.append(s_target_chunk_cpu)

                    if not step_s_input_chunks_cpu:
                        _log(f"    STEP路径: 文件 {file_path} 没有可处理的数据块。跳过此文件遍次内当前路径的训练。")
                    else:
                        # --- Cache Initialization (once per file pass for STEP path) ---
                        attn_indices_cfg_fp = config.use_attention_layer_indices or []
                        # Explicitly define LayerCacheS type for clarity if not already globally defined
                        # LayerCacheS = Tuple[Optional[MambaInferenceCache], Optional[Tuple[torch.Tensor, torch.Tensor]]]
                        batched_file_step_caches: List[Tuple[Optional[MambaInferenceCache], Optional[Tuple[torch.Tensor, torch.Tensor]]]] = []
                        for layer_idx_cache_init in range(config.n_layer):
                            is_attn_layer_cache_init = layer_idx_cache_init in attn_indices_cfg_fp
                            mamba_cache_init = None
                            attn_cache_init = None
                            if not is_attn_layer_cache_init: # Mamba layer
                                mamba_cache_init = MambaInferenceCache.alloc(
                                    batch_size=MODEL_PROCESSING_BATCH_SIZE_STEP,
                                    args=config, 
                                    device=DEVICE
                                )
                            else: # Attention layer
                                head_dim_init = config.d_model // config.num_attention_heads if config.num_attention_heads > 0 else 0
                                if head_dim_init > 0:
                                    past_k_init = torch.zeros(MODEL_PROCESSING_BATCH_SIZE_STEP, config.num_attention_heads, 0, head_dim_init, device=DEVICE)
                                    past_v_init = torch.zeros(MODEL_PROCESSING_BATCH_SIZE_STEP, config.num_attention_heads, 0, head_dim_init, device=DEVICE)
                                    attn_cache_init = (past_k_init, past_v_init)
                                else: 
                                    attn_cache_init = (torch.empty(0, device=DEVICE), torch.empty(0, device=DEVICE))
                            batched_file_step_caches.append((mamba_cache_init, attn_cache_init))
                        # --- End Cache Initialization ---

                        # Accumulators for gradient accumulation and logging
                        accumulated_loss_for_backward_step = torch.tensor(0.0, device=DEVICE)
                        num_j_iters_in_current_accumulation_cycle = 0 # Renamed from j_idx to represent batches of chunks
                        total_loss_tracker_for_file_pass_step = 0.0
                        num_loss_contributions_for_file_pass_step = 0

                        num_total_chunks_in_file = len(step_s_input_chunks_cpu)
                        num_batches_for_file_step = (num_total_chunks_in_file + MODEL_PROCESSING_BATCH_SIZE_STEP - 1) // MODEL_PROCESSING_BATCH_SIZE_STEP

                        chunk_pbar_step = tqdm(range(num_batches_for_file_step),
                                               desc=f"    STEP 文件 {file_idx+1}/{len(current_epoch_train_files)} 第 {pass_num_in_file + 1} 遍 批次",
                                               leave=False, position=1, bar_format=TQDM_BAR_FORMAT)

                        for batch_idx_in_file in chunk_pbar_step:
                            batch_start_chunk_idx = batch_idx_in_file * MODEL_PROCESSING_BATCH_SIZE_STEP
                            batch_end_chunk_idx = min(num_total_chunks_in_file, (batch_idx_in_file + 1) * MODEL_PROCESSING_BATCH_SIZE_STEP)
                            
                            current_s_input_batch_cpu_list_orig = [step_s_input_chunks_cpu[i] for i in range(batch_start_chunk_idx, batch_end_chunk_idx)]
                            current_s_target_batch_cpu_list_orig = [step_s_target_chunks_cpu[i] for i in range(batch_start_chunk_idx, batch_end_chunk_idx)]
                            current_actual_batch_size = len(current_s_input_batch_cpu_list_orig)

                            s_input_for_padding_op = list(current_s_input_batch_cpu_list_orig)
                            s_target_for_padding_op = list(current_s_target_batch_cpu_list_orig)

                            if current_actual_batch_size < MODEL_PROCESSING_BATCH_SIZE_STEP:
                                num_padding_items = MODEL_PROCESSING_BATCH_SIZE_STEP - current_actual_batch_size
                                dummy_chunk = torch.full((CHUNK_MAX_TIME_STEPS_STEP, config.dac_num_quantizers), 
                                                         EOS_TOKEN_ID, dtype=torch.long, device=CPU_DEVICE) # Create on CPU
                                for _ in range(num_padding_items):
                                    s_input_for_padding_op.append(dummy_chunk)
                                    s_target_for_padding_op.append(dummy_chunk) # Target dummy also EOS, but will be ignored by loss due to PAD_TOKEN_S_VALUE in s_target_batch_padded_for_loss
                            
                            s_input_batch_padded = torch.nn.utils.rnn.pad_sequence(
                                s_input_for_padding_op, batch_first=True, padding_value=EOS_TOKEN_ID
                            ).to(DEVICE, non_blocking=True) 
                            s_target_batch_padded_for_loss = torch.nn.utils.rnn.pad_sequence(
                                s_target_for_padding_op, batch_first=True, padding_value=PAD_TOKEN_S_VALUE
                            ).to(DEVICE, non_blocking=True)
                            
                            current_batch_total_loss_sum_step = torch.tensor(0.0, device=DEVICE)
                            style_vector_for_batch_step = style_vector_this_file_gpu.expand(MODEL_PROCESSING_BATCH_SIZE_STEP, -1)
                            previous_output_s_tokens_batch = None 
                            actual_seq_len_for_batch = s_input_batch_padded.shape[1] # Should be CHUNK_MAX_TIME_STEPS_STEP

                            for t in range(actual_seq_len_for_batch):
                                if t > 0 and torch.rand(1).item() > epsilon_for_this_step_path and previous_output_s_tokens_batch is not None:
                                    s_input_t_model = previous_output_s_tokens_batch 
                                else:
                                    s_input_t_model = s_input_batch_padded[:, t, :] 
                                
                                s_target_t_actual_for_loss_step = s_target_batch_padded_for_loss[:, t, :]

                                output_logits_t, batched_file_step_caches = model.step(
                                    s_step_indices=s_input_t_model, 
                                    caches=batched_file_step_caches,
                                    style_vector=style_vector_for_batch_step 
                                )
                                output_logits_t_reshaped = output_logits_t.view(
                                    MODEL_PROCESSING_BATCH_SIZE_STEP, 1, config.dac_num_quantizers, config.dac_codebook_size
                                )

                                if epsilon_for_this_step_path < 1.0:
                                    previous_output_s_tokens_batch = torch.argmax(output_logits_t_reshaped, dim=-1).squeeze(1)
                                
                                output_logits_t_active = output_logits_t_reshaped[:current_actual_batch_size, 0, :, :]
                                s_target_t_active_for_loss_step = s_target_t_actual_for_loss_step[:current_actual_batch_size, :]

                                if current_actual_batch_size > 0:
                                    loss_t = loss_fn(
                                        output_logits_t_active.reshape(-1, config.dac_codebook_size),
                                        s_target_t_active_for_loss_step.reshape(-1)
                                    )
                                    if not (torch.isnan(loss_t) or torch.isinf(loss_t)):
                                        current_batch_total_loss_sum_step += loss_t
                            
                            if actual_seq_len_for_batch > 0 and current_actual_batch_size > 0:
                                avg_loss_for_this_chunk_batch = current_batch_total_loss_sum_step / actual_seq_len_for_batch
                                accumulated_loss_for_backward_step += avg_loss_for_this_chunk_batch
                                total_loss_tracker_for_file_pass_step += avg_loss_for_this_chunk_batch.item() * current_actual_batch_size 
                                num_loss_contributions_for_file_pass_step += current_actual_batch_size 
                                num_j_iters_in_current_accumulation_cycle += 1

                            chunk_pbar_step.set_postfix_str(f"AccLossBwd: {accumulated_loss_for_backward_step.item():.4f} (BatchesInCycle: {num_j_iters_in_current_accumulation_cycle}, FileLossItems: {num_loss_contributions_for_file_pass_step})", refresh=True)

                            # GRADIENT ACCUMULATION CHECK
                            is_accumulation_step = num_j_iters_in_current_accumulation_cycle >= GRAD_ACCUMULATION_STEPS
                            is_last_chunk_batch_for_file_pass = (batch_idx_in_file == num_batches_for_file_step - 1)

                            if (is_accumulation_step or is_last_chunk_batch_for_file_pass) and num_j_iters_in_current_accumulation_cycle > 0:
                                if accumulated_loss_for_backward_step.item() > 0 and \
                                   not (torch.isnan(accumulated_loss_for_backward_step) or torch.isinf(accumulated_loss_for_backward_step)):
                                    
                                    loss_to_backward_finally = accumulated_loss_for_backward_step / num_j_iters_in_current_accumulation_cycle
                                    
                                    # Always False for typical gradient accumulation after summing losses
                                    retain_graph_for_this_backward = False

                                    if USE_MIXED_PRECISION and DEVICE.type == 'cuda':
                                        scaler.scale(loss_to_backward_finally).backward(retain_graph=retain_graph_for_this_backward)
                                        if CLIP_GRAD_NORM > 0:
                                            scaler.unscale_(optimizer)
                                            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                                        scaler.step(optimizer)
                                        scaler.update()
                                    else:
                                        loss_to_backward_finally.backward(retain_graph=retain_graph_for_this_backward)
                                        if CLIP_GRAD_NORM > 0:
                                            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                                        optimizer.step()
                                    
                                    optimizer.zero_grad(set_to_none=True)
                                else:
                                    _log(f"DEBUG STEP: Skipping backward pass due to zero/NaN/Inf accumulated_loss_for_backward_step. val: {accumulated_loss_for_backward_step.item():.4f}")
                                    optimizer.zero_grad(set_to_none=True) # Still zero grads if backward was skipped

                                accumulated_loss_for_backward_step = torch.tensor(0.0, device=DEVICE)
                                num_j_iters_in_current_accumulation_cycle = 0
                                
                                # Detach caches here, before they are used in the next accumulation cycle for the same file pass
                                if not is_last_chunk_batch_for_file_pass:
                                    if batched_file_step_caches: # Ensure caches exist
                                        temp_detached_caches = []
                                        for m_cache, attn_cache in batched_file_step_caches:
                                            detached_m_cache_component = None
                                            if m_cache is not None: # MambaInferenceCache
                                                # Detach internal tensors of MambaInferenceCache IN-PLACE
                                                m_cache.conv_state.detach_() # Use .detach_() for in-place detach
                                                m_cache.ssm_state.detach_()  # Use .detach_() for in-place detach
                                                detached_m_cache_component = m_cache # Reuse the object with detached tensors
                                            
                                            detached_attn_cache_component = None
                                            if attn_cache is not None:
                                                past_k, past_v = attn_cache
                                                detached_attn_cache_component = (past_k.detach(), past_v.detach())
                                            
                                            temp_detached_caches.append((detached_m_cache_component, detached_attn_cache_component))
                                        batched_file_step_caches = temp_detached_caches
                            
                            # End of j_idx loop (batch of chunks processing)
                            avg_loss_this_file_pass = total_loss_tracker_for_file_pass_step / num_loss_contributions_for_file_pass_step if num_loss_contributions_for_file_pass_step > 0 else float('inf')
                            _log(f"  常规训练：文件 '{os.path.basename(file_path)}', 第 {pass_num_in_file + 1} 遍完成。此遍平均主损失: {avg_loss_this_file_pass:.4f}")

            global_files_processed_counter += 1
            _log(f"  常规训练：文件 '{os.path.basename(file_path)}' (全局 #{global_files_processed_counter}) 所有 {NUM_PASSES_PER_FILE_IN_EPOCH} 遍处理完毕。")
            
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            safe_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_filename)
            current_save_dir = os.path.join(STEP_MODEL_SAVE_DIR, f"checkpoint_file_{global_files_processed_counter}_{safe_filename}")
            os.makedirs(current_save_dir, exist_ok=True)
            model.save_pretrained(current_save_dir)
            _log(f"  模型已在处理完文件 (全局 #{global_files_processed_counter}) 后保存到 {current_save_dir}")
            
            if MAX_RECENT_CHECKPOINTS_TO_KEEP > 0:
                manage_checkpoints(STEP_MODEL_SAVE_DIR, MAX_RECENT_CHECKPOINTS_TO_KEEP, _log)
            
            scheduler.step()

            if TEACHER_FORCING_EPOCH_INTERVAL > 0 and global_files_processed_counter > 0 and \
               global_files_processed_counter % TEACHER_FORCING_EPOCH_INTERVAL == 0:
                _log(f"\n--- 在全局文件 #{global_files_processed_counter} 后触发教师强制周期 ---")
                # ... (教师强制周期的实现将在此处，类似于 train_autoregressive_step.py 的 TF 块)
                # 记住调整输入 (S 而不是 ZQ)、风格计算和损失。
                _log(f"--- 教师强制周期结束 ---")

            if VALIDATION_INTERVAL_FILES > 0 and validation_files and \
               global_files_processed_counter > 0 and \
               global_files_processed_counter % VALIDATION_INTERVAL_FILES == 0:
                
                avg_val_loss = run_validation_pass(
                    model=model,
                    config=config,
                    validation_files_list=validation_files,
                    loss_fn=loss_fn,
                    _log_func=_log,
                    current_epoch_num_for_log=epoch + 1,
                    device=DEVICE,
                    cpu_device=CPU_DEVICE,
                    chunk_max_time_steps=CHUNK_MAX_TIME_STEPS_STEP,
                    model_step_processing_batch_size=MODEL_PROCESSING_BATCH_SIZE_STEP,
                    eos_token_id_for_padding_or_target=EOS_TOKEN_ID,
                    pad_token_s_value_for_loss=PAD_TOKEN_S_VALUE
                )
                if avg_val_loss < best_avg_val_loss:
                    best_avg_val_loss = avg_val_loss
                    best_model_save_path = os.path.join(STEP_MODEL_SAVE_DIR, BEST_MODEL_SUBDIR)
                    os.makedirs(best_model_save_path, exist_ok=True)
                    model.save_pretrained(best_model_save_path)
                    _log(f"*** 新的最佳模型已保存到 {best_model_save_path} (验证损失: {best_avg_val_loss:.4f}) ***")

        avg_epoch_main_loss = epoch_total_main_loss / num_loss_contributions_in_epoch if num_loss_contributions_in_epoch > 0 else float('inf')
        _log(f"--- 常规周期 {epoch + 1} 完成。平均主训练损失: {avg_epoch_main_loss:.4f} --- ")
        current_epoch_lr = optimizer.param_groups[0]['lr'] 
        _log(f"--- 常规周期 {epoch + 1} 结束时学习率: {current_epoch_lr:.2e} ---")
        
    _log(f"\n--- 所有训练周期完成 ({TOTAL_TRAINING_EPOCHS} 个周期) ---")
    log_f.close()

# --- Validation Pass Function (Adapted for S-in-S-out model) ---
def run_validation_pass(
    model: JambaForAudioGeneration_S_In_S_Out, 
    config: JambaSConfig,                    
    validation_files_list: List[str],
    loss_fn: nn.CrossEntropyLoss,
    _log_func,
    current_epoch_num_for_log: int,
    device: torch.device,
    cpu_device: torch.device,
    chunk_max_time_steps: int,
    model_step_processing_batch_size: int,
    eos_token_id_for_padding_or_target: int, 
    pad_token_s_value_for_loss: int         
):
    if not validation_files_list:
        _log_func("  验证：验证文件列表为空。跳过。")
        return float('inf')

    _log_func(f"--- 开始验证过程 (与周期 {current_epoch_num_for_log} 相关, {len(validation_files_list)} 个文件) ---")
    model.eval()
    total_val_main_loss = 0.0
    num_val_loss_contributions = 0

    with torch.no_grad():
        for val_file_idx, val_file_path in enumerate(tqdm(validation_files_list, desc=f"验证文件 (周期 {current_epoch_num_for_log})", unit="验证文件")):
            s_full_single_val_cpu: Optional[torch.Tensor] = load_s_tensor_from_file(val_file_path, cpu_device)
            if s_full_single_val_cpu is None:
                _log_func(f"    验证警告：从 {val_file_path} 加载 's' 失败，跳过。")
                continue

            s_originals_style_list_val = [s_full_single_val_cpu.to(model.get_style_encoder().device)]
            lengths_style_val = torch.tensor([s_full_single_val_cpu.shape[0]], dtype=torch.long, device=model.get_style_encoder().device)
            
            val_style_vector_gpu: Optional[torch.Tensor] = None
            try:
                val_style_vector_gpu = model.get_style_encoder()(s_originals_style_list_val, lengths_style_val)
                val_style_vector_gpu = val_style_vector_gpu.to(device)
            except Exception as e_style_val:
                _log_func(f"    验证错误：为 {val_file_path} 计算风格时出错: {e_style_val}。跳过。")
                continue
            if val_style_vector_gpu is None: continue

            s_input_val_full = s_full_single_val_cpu[:-1, :]
            s_target_val_full = s_full_single_val_cpu[1:, :]
            eos_row_val = torch.full((1, config.dac_num_quantizers), eos_token_id_for_padding_or_target, dtype=s_target_val_full.dtype, device=cpu_device)
            s_target_val_full_eos = torch.cat((s_target_val_full, eos_row_val), dim=0)

            val_s_input_chunks = []
            val_s_target_chunks = []
            total_steps_val_chunks = s_input_val_full.shape[0]

            for cs_idx in range(0, total_steps_val_chunks, chunk_max_time_steps):
                ce_idx = min(cs_idx + chunk_max_time_steps, total_steps_val_chunks)
                s_in_chk = s_input_val_full[cs_idx:ce_idx, :]
                s_targ_chk = s_target_val_full_eos[cs_idx:ce_idx, :]
                if s_in_chk.shape[0] == 0: continue
                val_s_input_chunks.append(s_in_chk)
                val_s_target_chunks.append(s_targ_chk)
            
            if not val_s_input_chunks: continue

            num_val_chunk_batches = (len(val_s_input_chunks) + model_step_processing_batch_size - 1) // model_step_processing_batch_size
            
            for j_val_idx in tqdm(range(num_val_chunk_batches), desc=f"  验证文件 {val_file_idx+1} 数据块", unit="验证块批次", leave=False):
                j_val_start = j_val_idx * model_step_processing_batch_size
                s_in_batch_cpu_val = val_s_input_chunks[j_val_start : j_val_start + model_step_processing_batch_size]
                s_targ_batch_cpu_val = val_s_target_chunks[j_val_start : j_val_start + model_step_processing_batch_size]

                if not s_in_batch_cpu_val: continue

                s_in_batch_pad_val, _ = custom_pad_s_sequences(s_in_batch_cpu_val, device, padding_value=eos_token_id_for_padding_or_target, batch_first=True)
                s_targ_batch_pad_val, _ = custom_pad_s_sequences(s_targ_batch_cpu_val, device, padding_value=pad_token_s_value_for_loss, batch_first=True)

                actual_b_size_val = s_in_batch_pad_val.shape[0]
                max_len_val_batch = s_in_batch_pad_val.shape[1]
                style_vec_batch_val = val_style_vector_gpu.repeat(actual_b_size_val, 1)

                val_caches: List[Tuple[Optional[MambaInferenceCache], Optional[Tuple[torch.Tensor, torch.Tensor]]]] = []
                attn_indices_val_cfg = config.use_attention_layer_indices or []
                for l_idx_val in range(config.n_layer):
                    is_attn_val = l_idx_val in attn_indices_val_cfg
                    m_cache_val = MambaInferenceCache.alloc(actual_b_size_val, config, device=device) if not is_attn_val else None
                    a_cache_val = None
                    val_caches.append((m_cache_val, a_cache_val))
                
                collected_logits_val = []
                s_current_step_input_val = torch.empty_like(s_in_batch_pad_val[:, 0:1, :])

                for t_val in range(max_len_val_batch):
                    s_current_step_input_val = s_in_batch_pad_val[:, t_val:t_val+1, :]
                    logits_step_val, val_caches = model.step(
                        s_step_indices=s_current_step_input_val,
                        caches=val_caches,
                        style_vector=style_vec_batch_val
                    )
                    collected_logits_val.append(logits_step_val)
                
                all_logits_val = torch.cat(collected_logits_val, dim=1)
                logits_for_loss_val = all_logits_val.reshape(-1, config.dac_codebook_size)
                target_for_loss_val = s_targ_batch_pad_val.reshape(-1)
                val_loss_this_batch = loss_fn(logits_for_loss_val, target_for_loss_val)

                if not (torch.isnan(val_loss_this_batch) or torch.isinf(val_loss_this_batch)):
                    total_val_main_loss += val_loss_this_batch.item() * actual_b_size_val
                    num_val_loss_contributions += actual_b_size_val
                else:
                    _log_func(f"    验证警告：验证文件 {val_file_idx+1}，块批次 {j_val_idx+1} 中出现 NaN/Inf 损失。")
    
    avg_val_loss = total_val_main_loss / num_val_loss_contributions if num_val_loss_contributions > 0 else float('inf')
    _log_func(f"--- 验证过程完成 (周期 {current_epoch_num_for_log} 相关)。平均验证主损失: {avg_val_loss:.4f} (基于 {num_val_loss_contributions} 个贡献样本) ---")
    model.train()
    return avg_val_loss

if __name__ == "__main__":
    train_model_step_by_step() 