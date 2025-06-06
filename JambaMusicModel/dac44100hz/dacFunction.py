import torch
import torchaudio
import os # Import os for path joining
# Import DAC directly from your local model.py
# from model import DAC
# from audiotools import AudioSignal
# Use direct import assuming dac44khz is a subdir relative to the script's location
# Need DAC for instantiation and DACConfig for loading config
from .dac44khz.model import DAC, DACConfig
# Need safetensors library for manual loading
from safetensors.torch import load_file 

# --- 修改后的加载函数 (手动加载) ---
def load_dac_model_from_safetensors(model_dir: str, target_device: torch.device):
    """
    从指定目录手动加载 DAC 模型，使用 config.json 和 model.safetensors 文件。

    Args:
        model_dir: 包含 config.json 和 model.safetensors 的目录路径。
        target_device: 要将模型加载到的设备 (例如 'cuda' 或 'cpu')。

    Returns:
        加载好的 DAC 模型。

    Raises:
        FileNotFoundError: 如果 config.json 或 model.safetensors 未找到。
        Exception: 如果加载过程中出现其他错误 (包括可能的依赖下载错误)。
    """
    config_path = os.path.join(model_dir, 'config.json')
    safetensors_path = os.path.join(model_dir, 'model.safetensors')
    
    print(f"尝试从目录 '{model_dir}' 手动加载 DAC 模型...")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"错误：在 '{model_dir}' 中找不到 config.json")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"错误：在 '{model_dir}' 中找不到 model.safetensors")

    try:
        # 1. 加载配置
        print(f"正在从 '{config_path}' 加载配置...")
        config = DACConfig.from_pretrained(model_dir) # 仍然推荐用 from_pretrained 加载 config

        # 2. 使用配置实例化模型骨架
        # !!! 警告: 这一步仍然可能触发 DAC.__init__ 中的 dac.utils.download !!!
        # 如果底层 dac 依赖未缓存且网络有问题，这里可能报错
        print("正在根据配置实例化模型...")
        model = DAC(config)

        # 3. 加载 .safetensors 文件中的权重
        print(f"正在从 '{safetensors_path}' 加载权重...")
        state_dict = load_file(safetensors_path, device="cpu") # 先加载到 CPU 以节省 GPU 内存

        # 4. 将权重加载到模型实例中
        print("正在将权重应用到模型...")
        model.load_state_dict(state_dict)
        
        # 5. 移动到目标设备并设置模式
        model.to(target_device)
        model.eval() 
        print("模型手动加载成功！")
        return model
        
    except ImportError:
        print("错误：请确保安装了 safetensors 库 (pip install safetensors)")
        raise
    except Exception as e:
        print(f"手动加载模型时出错: {e}")
        # 再次提醒：错误很可能来自 DAC.__init__ 中的 dac 依赖下载
        raise # 重新抛出异常


# --- 新增：保存编码张量的函数 ---
def save_encoded_tensors(zq: torch.Tensor, s: torch.Tensor, save_path: str):
    """
    将 DAC 编码产生的 zq 和 s 张量保存到文件。

    Args:
        zq: zq 张量。
        s: s 张量。
        save_path: 保存文件的完整路径 (例如 'encoded_output/encoded_audio.pt')。
    """
    try:
        # 确保目录存在
        output_dir = os.path.dirname(save_path)
        if output_dir: # 只有当路径包含目录时才创建
            os.makedirs(output_dir, exist_ok=True)

        # 将 zq 和 s 放入字典
        encoded_data = {
            'zq': zq.cpu(), # 移动到 CPU 保存
            's': s.cpu()    # 移动到 CPU 保存
        }

        # 保存字典到 .pt 文件
        torch.save(encoded_data, save_path)
        print(f"编码结果已成功保存到: {save_path}")
        return True
    except Exception as e:
        print(f"保存编码结果时出错: {e}")
        return False

# --- 新增：加载编码张量的函数 ---
def load_encoded_tensors(load_path: str, target_device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    从文件加载之前保存的 zq 和 s 张量。

    Args:
        load_path: 保存编码结果的文件路径。
        target_device: 要将张量加载到的目标设备。

    Returns:
        一个包含 (zq, s) 的元组。如果加载失败，则返回 (None, None)。
    """
    try:
        if os.path.exists(load_path):
            print(f"正在从 '{load_path}' 加载编码结果...")
            # map_location 参数确保张量被加载到正确的设备上，或者先加载到 CPU 再手动移动
            loaded_data = torch.load(load_path, map_location='cpu', weights_only=True) # 推荐先加载到 CPU

            # 从字典中取出 zq 和 s
            zq_loaded = loaded_data['zq'].to(target_device) # 移动到目标设备
            s_loaded = loaded_data['s'].to(target_device)   # 移动到目标设备

            print("编码结果加载成功。")
            print("zq_loaded shape:", zq_loaded.shape)
            print("s_loaded shape:", s_loaded.shape)
            return zq_loaded, s_loaded
        else:
            print(f"错误：找不到已保存的编码文件 {load_path}")
            return None, None
    except Exception as e:
        print(f"加载编码结果时出错: {e}")
        return None, None

# --- 新增：将目录中所有音频文件转换为张量并保存的函数 ---
def convert_audio_files_to_tensors(input_audio_dir: str, output_tensor_dir: str, model):
    """
    将指定输入目录下的所有音频文件转换为 DAC 编码的 zq 和 s 张量并保存在指定的输出目录中。
    实现断点续传功能：如果目标 .pt 文件已存在，则跳过转换。
    成功转换并保存一个文件后，此函数本身不删除源文件，而是返回成功处理的源文件路径列表。

    Args:
        input_audio_dir: 包含音频文件的输入目录路径。
        output_tensor_dir: 保存转换后张量 (.pt 文件) 的输出目录路径。
        model: 已加载的 DAC 模型实例。

    Returns:
        list[str]: 一个包含在此次运行中成功转换并保存的源音频文件完整路径的列表。
                   如果发生严重错误或没有文件被处理，可能返回空列表或 None。
    """
    print(f"开始处理目录 '{input_audio_dir}' 中的音频文件...")
    print(f"转换后的张量将保存到 '{output_tensor_dir}'。")

    if not os.path.isdir(input_audio_dir):
        print(f"错误：输入目录 '{input_audio_dir}' 不存在或不是一个目录。")
        return None # Indicate critical error

    try:
        os.makedirs(output_tensor_dir, exist_ok=True)
    except OSError as e:
        print(f"错误：无法创建输出目录 '{output_tensor_dir}': {e}")
        return None # Indicate critical error

    successfully_converted_source_paths = []
    processed_in_this_run_count = 0
    skipped_due_to_existing_pt = 0
    skipped_due_to_error_or_type = 0
    
    try:
        all_items_in_dir = os.listdir(input_audio_dir)
    except OSError as e:
        print(f"错误：无法读取输入目录 '{input_audio_dir}': {e}")
        return None

    print(f"在输入目录 '{input_audio_dir}' 中找到 {len(all_items_in_dir)} 个项目。开始处理...")

    # 将 supported_extensions 的定义移到循环外部以提高效率
    supported_extensions = ('.wav', '.flac', '.mp3', '.aac', '.ogg', '.m4a')

    for filename in all_items_in_dir:
        input_file_path = os.path.join(input_audio_dir, filename)

        if not os.path.isfile(input_file_path):
            # print(f"  跳过 '{filename}'，因为它是一个目录。") # Minor log, can be enabled if needed
            continue

        base, input_ext = os.path.splitext(filename)
        # 支持常见的音频格式，可根据需要扩展
        # supported_extensions = ('.wav', '.flac', '.mp3', '.aac', '.ogg', '.m4a') # 此行已移出循环
        if not input_ext.lower().endswith(supported_extensions):
            # print(f"  跳过 '{filename}': 文件扩展名 '{input_ext}' 不是支持的音频类型。")
            skipped_due_to_error_or_type +=1
            continue

        output_pt_filename = f"{base}.pt"
        output_pt_file_path = os.path.join(output_tensor_dir, output_pt_filename)

        # 断点续传核心逻辑: 检查 .pt 文件是否已存在
        if os.path.exists(output_pt_file_path):
            # print(f"  跳过 '{filename}': 目标 .pt 文件 '{output_pt_file_path}' 已存在。")
            skipped_due_to_existing_pt += 1
            continue

        try:
            # print(f"  正在处理 '{filename}'...")
            # 使用模型编码 (model.encode 通常期望路径)
            zq, s = model.encode(input_file_path)
            # print(f"    成功编码 '{filename}'. zq shape: {zq.shape}, s shape: {s.shape}")

            # 使用 save_encoded_tensors 函数保存 zq 和 s
            save_successful = save_encoded_tensors(zq, s, output_pt_file_path)

            if save_successful:
                # print(f"  已将 '{filename}' 的编码张量保存到: '{output_pt_file_path}'")
                successfully_converted_source_paths.append(input_file_path)
                processed_in_this_run_count += 1
            else:
                print(f"  警告：保存 '{filename}' 的编码张量失败 (save_encoded_tensors 返回 False)。")
                skipped_due_to_error_or_type += 1
                
        except RuntimeError as e:
            print(f"  处理文件 '{filename}' 时发生运行时错误 (可能是无效音频或模型问题): {e}")
            skipped_due_to_error_or_type += 1
        except Exception as e:
            print(f"  处理文件 '{filename}' 时发生意外错误: {e}")
            skipped_due_to_error_or_type += 1
    
    print(f"\n--- 目录 '{input_audio_dir}' 转 '{output_tensor_dir}' 处理总结 ---")
    print(f"在输入目录中找到的总项目数: {len(all_items_in_dir)}")
    print(f"本次运行中成功转换并保存的文件数: {processed_in_this_run_count}")
    print(f"跳过的文件数 (因目标 .pt 文件已存在): {skipped_due_to_existing_pt}")
    print(f"跳过的文件数 (因错误、非支持类型或保存失败): {skipped_due_to_error_or_type}")
    
    return successfully_converted_source_paths

def get_extracted_codebooks(model):
    if model:
        underlying_dac_model = model.dac  # 这是 descript-audio-codec 的模型实例

        if hasattr(underlying_dac_model, 'quantizer'):
            quantizer = underlying_dac_model.quantizer
            print(f"成功访问 quantizer，类型为: {type(quantizer)}")
            print(f"量化器 (quantizer) 的属性: {dir(quantizer)}")

            extracted_codebooks = []

            # 常见模式 1: 量化器有一个名为 'quantizers' 或 'vq_layers' 或 'layers' 的列表 (nn.ModuleList)
            # 这里的 'quantizers_list_attribute_name' 需要你根据 dir() 的输出确定
            # 可能是 'quantizers', 'vq', 'layers' 等

            # 尝试猜测一些常见的属性名
            possible_list_attrs = ['quantizers', 'vq_layers', 'layers']
            quantizer_list_found = False

            for attr_name in possible_list_attrs:
                if hasattr(quantizer, attr_name):
                    potential_list = getattr(quantizer, attr_name)
                    if isinstance(potential_list, torch.nn.ModuleList) or isinstance(potential_list, list):
                        print(f"找到可能是量化器列表的属性: '{attr_name}'，包含 {len(potential_list)} 个元素。")
                        quantizer_list_found = True
                        for i, individual_vq_layer in enumerate(potential_list):
                            print(f"  检查第 {i} 个单独量化层，类型: {type(individual_vq_layer)}")
                            print(f"    该层的属性: {dir(individual_vq_layer)}")

                            # 尝试寻找码本，通常在 'codebook.weight' 或 '_codebook.embed'
                            codebook_tensor = None
                            if hasattr(individual_vq_layer, 'codebook') and \
                                    isinstance(individual_vq_layer.codebook, torch.nn.Embedding) and \
                                    hasattr(individual_vq_layer.codebook, 'weight'):
                                codebook_tensor = individual_vq_layer.codebook.weight.data.clone().detach()
                                print(f"    从 'codebook.weight' 提取到码本 {i}，形状: {codebook_tensor.shape}")
                            elif hasattr(individual_vq_layer, '_codebook'):  # 类似于 EnCodec 的结构
                                if hasattr(individual_vq_layer._codebook, 'embed'):
                                    embed_obj = individual_vq_layer._codebook.embed
                                    if isinstance(embed_obj, torch.Tensor):
                                        codebook_tensor = embed_obj.clone().detach()
                                    elif isinstance(embed_obj, list) and len(embed_obj) > 0 and isinstance(embed_obj[0],
                                                                                                           torch.Tensor):
                                        codebook_tensor = embed_obj[0].clone().detach()  # 通常 embed 是包含一个张量的列表
                                    if codebook_tensor is not None:
                                        print(f"    从 '_codebook.embed' 提取到码本 {i}，形状: {codebook_tensor.shape}")

                            # 新增：提取 out_proj 权重和偏置
                            out_proj_weight_g = None # Renamed from out_proj_weight
                            out_proj_weight_v = None # New
                            out_proj_bias = None
                            if hasattr(individual_vq_layer, 'out_proj') and \
                               isinstance(individual_vq_layer.out_proj, torch.nn.Conv1d):
                                out_proj_layer = individual_vq_layer.out_proj
                                # print(f"    属性列表 for out_proj_layer (type: {type(out_proj_layer)}): {dir(out_proj_layer)}") 

                                if hasattr(out_proj_layer, 'weight_g') and hasattr(out_proj_layer, 'weight_v'):
                                    out_proj_weight_g = out_proj_layer.weight_g.data.clone().detach()
                                    out_proj_weight_v = out_proj_layer.weight_v.data.clone().detach()
                                    print(f"    从 'out_proj' 提取到 weight_g 形状: {out_proj_weight_g.shape}, weight_v 形状: {out_proj_weight_v.shape}")
                                else:
                                    print(f"    警告: out_proj_layer (Conv1d) 缺少 weight_g 或 weight_v。将回退到提取 .weight。")
                                    # Fallback for safety, though ideally this path isn't taken if WN is standard
                                    out_proj_weight_g = out_proj_layer.weight.data.clone().detach() # Store in g for now
                                    out_proj_weight_v = None # Indicate v is not separately available
                                    print(f"    回退：从 'out_proj' 提取到 .weight (存为g) 形状: {out_proj_weight_g.shape}")

                                if out_proj_layer.bias is not None:
                                    out_proj_bias = out_proj_layer.bias.data.clone().detach()
                                    if out_proj_weight_g is not None: # Ensure we also print bias shape if g/v were found
                                        print(f"      同时提取到偏置形状: {out_proj_bias.shape}")
                            else:
                                print(f"    警告：未能从第 {i} 层提取 'out_proj' 层或它不是 Conv1d。")

                            if codebook_tensor is not None and out_proj_weight_g is not None:
                                extracted_codebooks.append({
                                    'codebook': codebook_tensor,
                                    'out_proj_weight_g': out_proj_weight_g,
                                    'out_proj_weight_v': out_proj_weight_v, # Can be None if fallback used
                                    'out_proj_bias': out_proj_bias
                                })
                            else:
                                print(f"    未能完整提取第 {i} 层的码本和/或必要的投影层。")
                        break 

            if not quantizer_list_found:
                print(f"未能自动找到量化器列表。请仔细查看 'dir(quantizer)' 的输出，寻找包含各个量化阶段的列表或模块。")
                print(f"例如，如果看到 'vq' 属性，尝试检查 'dir(quantizer.vq)'，它内部可能还有 'layers'。")

            if extracted_codebooks:
                print(f"\n成功提取了 {len(extracted_codebooks)} 个码本。")
                # return extracted_codebooks
                # 你可以将这些码本保存到文件
                torch.save(extracted_codebooks, "extracted_codebooks.pth")
                # print("码本已保存到 extracted_codebooks.pth")
            else:
                print("\n未能提取任何码本。需要根据上面 dir() 的输出进行手动分析和调整代码。")
                print(
                    "      例如，如果 quantizer 有一个 'vq' 属性，那么实际的码本可能在 quantizer.vq.layers[i].codebook.weight。")
                return

        else:
            print(f"错误: 'model.dac' (类型: {type(underlying_dac_model)}) 中没有名为 'quantizer' 的属性。")
            print(f"     'model.dac' 的可用属性: {dir(underlying_dac_model)}")
    else:
        print("模型未能加载，无法提取码本。")

def s_to_zq_custom(s: torch.Tensor, codebooks_path: str, target_device: torch.device) -> torch.Tensor | None:
    """
    使用从指定路径加载的码本和投影层将编码索引 s 转换为 zq。
    这是对 descript-audio-codec 中 quantizer.from_codes() 方法的复现。

    Args:
        s: 编码索引张量，形状应为 (batch_size, num_codebooks, sequence_length)。
           s 中的值是对应码本的索引。
        codebooks_path: 保存码本和投影层参数列表的 .pth 文件路径。
        target_device: 计算结果的目标设备。

    Returns:
        zq 张量，形状 (batch_size, projected_dim, sequence_length)，如果加载或处理失败则返回 None。
    """
    try:
        # weights_only=True is safer if you trust the source of the .pth file.
        # For this internal use case, False is fine if the file is generated by this script.
        codebooks_data_list = torch.load(codebooks_path, map_location='cpu', weights_only=False) 
        if not isinstance(codebooks_data_list, list):
            print(f"错误：从 '{codebooks_path}' 加载的码本数据不是一个列表。")
            return None
        if not codebooks_data_list:
            print(f"错误：从 '{codebooks_path}' 加载的码本列表为空。")
            return None
        if not all(isinstance(item, dict) for item in codebooks_data_list):
            print(f"错误：码本列表中的元素并非都是字典。")
            return None
        if not all('codebook' in item and 'out_proj_weight_g' in item for item in codebooks_data_list):
            print(f"错误：码本字典中缺少 'codebook' 或 'out_proj_weight_g'。")
            return None
        # print(f"成功从 '{codebooks_path}' 加载了 {len(codebooks_data_list)} 个码本层的数据。")
    except FileNotFoundError:
        print(f"错误：找不到码本文件 '{codebooks_path}'。")
        return None
    except Exception as e:
        print(f"加载码本文件 '{codebooks_path}' 时出错: {e}")
        return None

    if s.shape[1] != len(codebooks_data_list):
        print(f"错误：s 张量的第二维 ({s.shape[1]}) 与加载的码本层数量 ({len(codebooks_data_list)}) 不匹配。")
        return None

    s = s.to(target_device)
    batch_size, num_quantizers, seq_len = s.shape

    # 确定输出维度 (projected_dim)
    # 假设所有 out_proj 层输出到相同的维度，并且 kernel_size=1
    # out_proj_weight_g shape: (out_channels, 1, 1) or (out_channels) for broadcasting
    # out_proj_weight_v shape: (out_channels, in_channels, kernel_width)
    first_out_proj_g = codebooks_data_list[0]['out_proj_weight_g']
    if not isinstance(first_out_proj_g, torch.Tensor):
        print("错误：第一个码本层的 'out_proj_weight_g' 不是张量。")
        return None
    projected_dim = first_out_proj_g.shape[0] # out_channels from weight_g

    zq = torch.zeros(batch_size, projected_dim, seq_len, device=target_device)
    eps = 1e-12 # For numerical stability in normalization, similar to F.normalize default if not specified

    for i in range(num_quantizers):
        current_layer_data = codebooks_data_list[i]
        
        current_codebook_tensor = current_layer_data['codebook'].to(target_device)
        w_g = current_layer_data['out_proj_weight_g'].to(target_device)
        w_v = current_layer_data['out_proj_weight_v'] # Might be None
        bias = current_layer_data['out_proj_bias'].to(target_device) if current_layer_data['out_proj_bias'] is not None else None
        
        indices_for_current_codebook = s[:, i, :]  # Shape: (batch_size, sequence_length)
        embedded_vectors = torch.nn.functional.embedding(indices_for_current_codebook, current_codebook_tensor)
        permuted_embedded_vectors = embedded_vectors.permute(0, 2, 1)

        effective_weight = None
        if w_v is not None: # We have g and v, so reconstruct normalized weight
            w_v = w_v.to(target_device)
            # Normalize w_v: (O, I, K) -> norm over dim 1 (I) and 2 (K)
            # For Conv1d with kernel_size=1, K=1. So norm over dim=1 (in_channels)
            # norm = w_v.norm(p=2, dim=1, keepdim=True) # This would be for when K might be > 1
            # More precisely for Conv1d weights (O,I,K), normalization is over dims (1,2)
            # For kernel_size=1, this means norm over (I,1) effectively. PyTorch F.normalize handles this.
            # The `dim` for F.normalize should be the dimension along which to compute the norm for each slice.
            # For weight_norm on Conv1d, it standardly normalizes each filter vector.
            # weight_v is (out_channels, in_channels, kernel_size=1)
            # We want to normalize each (in_channels, 1) vector for each out_channel.
            # Or, more simply, F.normalize(w_v, p=2, dim=1) if w_g broadcasts correctly. Let's check shapes.
            # w_g is often (out_channels) or (out_channels, 1, 1)
            # w_v is (out_channels, in_channels, 1)
            # The norm should be computed for each of the `out_channels` filters, across their `in_channels * kernel_size` elements.
            # So, for each g_i, v_i, effective_w_i = g_i * (v_i / ||v_i||)
            # ||v_i|| is norm of v[i, :, :] which is (in_channels, kernel_size)
            
            # Standard way with torch.nn.utils.weight_norm applied to Conv1d:
            # computes norm over all dimensions of v except the first (out_channels)
            # So for v of shape (O, I, K), norm is computed over (I,K) for each O.
            norm_v = w_v.norm(p=2, dim=(1,2), keepdim=True) # (O, 1, 1)
            normalized_v = w_v / (norm_v + eps)
            effective_weight = w_g * normalized_v # w_g is (O,1,1) or (O) and should broadcast
        else: # Fallback: w_g actually contains the .weight directly
            effective_weight = w_g
        
        projected_vectors_for_layer = torch.nn.functional.conv1d(permuted_embedded_vectors, effective_weight, bias, stride=1, padding=0)
        
        zq += projected_vectors_for_layer

    return zq


if __name__ == '__main__':
    # --- 主脚本逻辑 ---
    model_path = './dac44khz'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 调用新的加载函数
    try:
        model = load_dac_model_from_safetensors(model_path, device)
    except Exception:
        print("无法完成模型加载，脚本退出。")
        exit()

    # model.

    # 提取码本的函数 (或者直接在此处执行提取逻辑)
    def get_extracted_codebooks_and_save(model_instance, save_path="extracted_codebooks.pth"):
        if model_instance:
            underlying_dac_model = model_instance.dac
            if hasattr(underlying_dac_model, 'quantizer'):
                quantizer = underlying_dac_model.quantizer
                print(f"成功访问 quantizer，类型为: {type(quantizer)}")
                if hasattr(quantizer, 'quantizer_dropout'):
                    print(f"量化器 dropout 率 (quantizer.quantizer_dropout): {quantizer.quantizer_dropout}")

                extracted_codebooks_list = []
                possible_list_attrs = ['quantizers', 'vq_layers', 'layers']
                quantizer_list_found = False

                for attr_name in possible_list_attrs:
                    if hasattr(quantizer, attr_name):
                        potential_list = getattr(quantizer, attr_name)
                        if isinstance(potential_list, torch.nn.ModuleList) or isinstance(potential_list, list):
                            print(f"找到可能是量化器列表的属性: '{attr_name}'，包含 {len(potential_list)} 个元素。")
                            quantizer_list_found = True
                            for i, individual_vq_layer in enumerate(potential_list):
                                print(f"  检查第 {i} 个单独量化层，类型: {type(individual_vq_layer)}")

                                codebook_tensor = None
                                if hasattr(individual_vq_layer, 'codebook') and \
                                        isinstance(individual_vq_layer.codebook, torch.nn.Embedding) and \
                                        hasattr(individual_vq_layer.codebook, 'weight'):
                                    codebook_tensor = individual_vq_layer.codebook.weight.data.clone().detach()
                                    print(f"    从 'codebook.weight' 提取到原始码本 {i}，形状: {codebook_tensor.shape}")
                                    
                                    # --- >>>> 新增：扩展码本以包含EOS <<<< ---
                                    if codebook_tensor is not None and codebook_tensor.shape[0] == 1024:
                                        eos_embedding = torch.zeros(1, codebook_tensor.shape[1], device=codebook_tensor.device, dtype=codebook_tensor.dtype)
                                        codebook_tensor = torch.cat((codebook_tensor, eos_embedding), dim=0)
                                        print(f"      已扩展码本 {i} 以包含EOS，新形状: {codebook_tensor.shape}")
                                    elif codebook_tensor is not None and codebook_tensor.shape[0] != 1024:
                                         print(f"      警告: 码本 {i} 的原始大小不是1024 (实际为 {codebook_tensor.shape[0]})，未进行EOS扩展。")
                                    # --- <<<< EOS扩展结束 <<<< ---
                                
                                out_proj_weight_g = None
                                out_proj_weight_v = None
                                out_proj_bias = None
                                if hasattr(individual_vq_layer, 'out_proj') and \
                                   isinstance(individual_vq_layer.out_proj, torch.nn.Conv1d):
                                    out_proj_layer = individual_vq_layer.out_proj

                                    if hasattr(out_proj_layer, 'weight_g') and hasattr(out_proj_layer, 'weight_v'):
                                        out_proj_weight_g = out_proj_layer.weight_g.data.clone().detach()
                                        out_proj_weight_v = out_proj_layer.weight_v.data.clone().detach()
                                        print(f"    从 'out_proj' 提取到 weight_g 形状: {out_proj_weight_g.shape}, weight_v 形状: {out_proj_weight_v.shape}")
                                    else:
                                        print(f"    警告: out_proj_layer (Conv1d) 缺少 weight_g 或 weight_v。将回退到提取 .weight。")
                                        out_proj_weight_g = out_proj_layer.weight.data.clone().detach()
                                        out_proj_weight_v = None
                                        print(f"    回退：从 'out_proj' 提取到 .weight (存为g) 形状: {out_proj_weight_g.shape}")

                                    if out_proj_layer.bias is not None:
                                        out_proj_bias = out_proj_layer.bias.data.clone().detach()
                                        if out_proj_weight_g is not None:
                                            print(f"      同时提取到偏置形状: {out_proj_bias.shape}")
                                else:
                                    print(f"    警告：未能从第 {i} 层提取 'out_proj' 层或它不是 Conv1d。")

                                if codebook_tensor is not None and out_proj_weight_g is not None:
                                    # 确保保存的 codebook_tensor 是可能已扩展的那个
                                    extracted_codebooks_list.append({
                                        'codebook': codebook_tensor, 
                                        'out_proj_weight_g': out_proj_weight_g,
                                        'out_proj_weight_v': out_proj_weight_v,
                                        'out_proj_bias': out_proj_bias
                                    })
                                else:
                                    print(f"    未能完整提取第 {i} 层的码本和/或必要的投影层。跳过此层。")
                            break 
                
                if not quantizer_list_found:
                    print(f"未能自动找到量化器列表。")

                if extracted_codebooks_list:
                    print(f"\n成功提取并处理了 {len(extracted_codebooks_list)} 个码本层的数据。")
                    torch.save(extracted_codebooks_list, save_path)
                    print(f"码本数据已保存到 {save_path}")
                else:
                    print("\n未能提取任何码本数据。")
            else:
                print(f"错误: 'model.dac' 中没有名为 'quantizer' 的属性。")
        else:
            print("模型未能加载，无法提取码本。")

    # 调用函数来提取并保存码本数据
    get_extracted_codebooks_and_save(model, "extracted_codebooks.pth")
    # get_extracted_codebooks(model) # 注释掉旧的调用

    # 创建一个示例的 s 张量
    # 现在模型和码本期望索引达到1024 (EOS), 所以randint上限是1025
    example_s = torch.randint(0, 1025, (1, 9, 100), device=device)
    codebooks_file_path = "extracted_codebooks.pth"

    # 使用自定义函数进行转换
    zq_custom_result = s_to_zq_custom(example_s, codebooks_file_path, device)
    
    if zq_custom_result is not None:
        print("自定义 s_to_zq 转换结果 zq_custom_result shape:", zq_custom_result.shape)

        if model and hasattr(model.dac, 'quantizer'):
            zq_original_result, _, _ = model.dac.quantizer.from_codes(example_s.to(model.device)) 
            print("原始模型 from_codes 结果 zq_original_result shape:", zq_original_result.shape)
            
            # 确保比较前在同一设备
            custom_res_device = zq_custom_result.to(model.device)
            orig_res_device = zq_original_result.to(model.device)

            print(f"自定义结果在 {custom_res_device.device}, 原始结果在 {orig_res_device.device}")

            # 更严格的比较，移除显式 atol (或使用默认的非常小的值)
            # PyTorch allclose default: rtol=1e-05, atol=1e-08
            if torch.allclose(custom_res_device, orig_res_device):
                print("自定义转换结果与原始模型结果在默认容忍度下一致！ (非常严格)")
            else:
                print("警告：自定义转换结果与原始模型结果在默认容忍度下不一致。")
                # 保留之前的 atol=1e-6 检查作为参考
                if torch.allclose(custom_res_device, orig_res_device, atol=1e-6):
                    print("自定义转换结果与原始模型结果在 atol=1e-6 下一致。")
                elif torch.allclose(custom_res_device, orig_res_device, atol=1e-4, rtol=1e-3):
                    print("自定义转换结果与原始模型结果在 atol=1e-4, rtol=1e-3 下一致。")
                else:
                    print("警告：自定义转换结果与原始模型结果即使在较大容忍度下仍不一致。")
                
                diff = torch.abs(custom_res_device - orig_res_device)
                print(f"最大差异: {diff.max().item()}, 平均差异: {diff.mean().item()}")
                
                # 检查是否存在 NaN
                if torch.isnan(custom_res_device).any():
                    print("警告: 自定义结果中存在 NaN 值。")
                if torch.isnan(orig_res_device).any():
                    print("警告: 原始结果中存在 NaN 值。")

    else:
        print("自定义 s_to_zq 转换失败。")

    # ... (wav_path 和后续的音频处理代码保持不变) ...
    wav_path = 'D:\\\\music\\\\input\\\\Designant. - Designant.flac'



