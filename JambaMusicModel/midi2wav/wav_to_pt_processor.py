import os
import torch
import time # 新增 time 模块导入
import json # 导入 json 模块
import traceback # 导入 traceback 模块
import torchaudio # 导入 torchaudio 用于加载音频
# from dac44100hz.dacFunction import load_dac_model_from_safetensors, convert_audio_files_to_tensors
# 修正的导入路径，假设 dac44100hz 是一个包或目录，与此脚本（如果此脚本在 for_trae/ 中）位于同一级别
# 如果 dacFunction.py 在 for_trae/ 内的 dac44100hz 目录中，则导入方式为：
from ..dac44100hz.dacFunction import load_dac_model_from_safetensors # Potentially remove convert_audio_files_to_tensors if unused


# --- 配置 ---
# 推断的 midi2wav.py 的输出路径 (Windows 格式)
DEFAULT_WAV_INPUT_DIR = 'F:/music/dataset/lmd_full.tar/lmd_full_wav'
# ubuntu
# DEFAULT_WAV_INPUT_DIR = '/root/autodl-tmp/modelTrain/wav'

# .pt 文件的建议输出目录 (Windows 格式)
DEFAULT_PT_OUTPUT_DIR = 'F:/music/dataset/lmd_full.tar/lmd_full_pt'
# ubuntu
# DEFAULT_PT_OUTPUT_DIR = '/root/autodl-tmp/modelTrain/jambaDataset2'

# DAC 模型的相对路径 (假设 dac44khz 文件夹在 dac44100hz 文件夹内)
DEFAULT_MODEL_PATH = '../dac44100hz/dac44khz'

# JSON 状态文件路径 (必须与 midi2wav.py 中的一致)
STATUS_JSON_FILENAME = 'processing_status.json'
# 从 DEFAULT_WAV_INPUT_DIR派生，以确保与 midi2wav.py 的一致性
# 这将类似于 'F:/music/dataset/lmd_full.tar/processing_status.json'
STATUS_JSON_PATH = os.path.join(os.path.dirname(os.path.normpath(DEFAULT_WAV_INPUT_DIR)), STATUS_JSON_FILENAME)

# --- 等待逻辑的新常量 ---
WAIT_INTERVAL_SECONDS = 20         # 当没有状态为0的文件时，扫描JSON的等待间隔

def load_status_from_json(json_path: str) -> dict:
    """从JSON文件加载状态数据。"""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                # Handle empty file case which is valid JSON for an empty object if midi2wav created it
                content = f.read()
                if not content.strip(): # If file is empty or only whitespace
                    print(f"信息: 状态JSON文件 '{json_path}' 为空。将视为空状态对象 {{}}。")
                    return {}
                return json.loads(content) # Use json.loads on the content string
        except json.JSONDecodeError as e_decode:
            print(f"警告: 无法解析JSON状态文件 '{json_path}' (JSONDecodeError): {e_decode}。将使用空状态。")
            return {}
        except OSError as e_os:
            print(f"警告: 无法读取JSON状态文件 '{json_path}' (OSError): {e_os}。将使用空状态。")
            return {}
    # If JSON file does not exist at startup for wav_to_pt, midi2wav.py should have created it.
    # If it's truly not there (e.g., midi2wav failed or paths are mismatched), treat as empty.
    print(f"信息: 状态JSON文件 '{json_path}' 未找到。将视为空状态。midi2wav.py 应该会创建此文件。")
    return {}

def save_status_to_json(json_path: str, status_data: dict):
    """将状态数据保存到JSON文件。"""
    try:
        # 确保JSON文件的目录存在
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=4)
    except OSError as e:
        print(f"错误: 无法写入JSON状态文件 '{json_path}': {e}")

def _process_single_wav_to_pt(
    wav_filepath: str,
    pt_output_dir: str,
    model,
    device: torch.device,
    expected_sr: int = 44100
) -> bool:
    """处理单个WAV文件到PT张量。返回True表示成功，False表示失败。"""
    wav_filename = os.path.basename(wav_filepath)
    base_name_no_ext = os.path.splitext(wav_filename)[0]
    pt_filename = base_name_no_ext + ".pt"
    pt_filepath = os.path.join(pt_output_dir, pt_filename)

    if not os.path.exists(wav_filepath):
        print(f"  错误: WAV文件在尝试转换前未找到: {wav_filepath}")
        return False

    if os.path.exists(pt_filepath):
        print(f"  信息: PT文件 '{pt_filepath}' 已存在。跳过转换。")
        return True

    try:
        print(f"  处理中: {wav_filepath} -> {pt_filepath}")

        # 音频加载和预处理已移除，假设 model.encode(wav_filepath) 会处理这些。
        # with torch.no_grad():
        #     encode_result_tuple = model.encode(wav_filepath)
        #
        #     if not (isinstance(encode_result_tuple, tuple) and len(encode_result_tuple) >= 2):
        #         print(f"    错误: 模型编码输出格式不正确: {type(encode_result_tuple)}. 预期是一个至少包含2个元素的元组。跳过 '{wav_filename}'.")
        #         return False
        #
        #     codes_tensor = encode_result_tuple[1]
        #     scale_tensor = None # DAC 的 .encode 通常不以此格式返回 scale
        #
        # data_to_save = {'codes': codes_tensor.cpu().numpy()}
        # if scale_tensor is not None:
        #     data_to_save['scale'] = scale_tensor.cpu().numpy()
        
        # 可选: data_to_save['_meta'] = {'sr': expected_sr, 'model_tag': 'dac_44khz'}

        with torch.no_grad():
            zq, s = model.encode(wav_filepath)

            data_to_save = {
                'zq': zq.cpu(),  # 移动到 CPU 保存
                's': s.cpu()  # 移动到 CPU 保存
            }

        torch.save(data_to_save, pt_filepath)
        print(f"    成功转换并保存到 {pt_filepath}")
        return True

    except Exception as e:
        print(f"    处理WAV文件 '{wav_filepath}' 时发生错误: {e}")
        traceback.print_exc()
        return False

def process_wav_to_pt_with_resume(
    wav_input_dir: str = DEFAULT_WAV_INPUT_DIR,
    pt_output_dir: str = DEFAULT_PT_OUTPUT_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    delete_source_wav_on_success: bool = True
):
    """
    将 WAV 文件从输入目录转换为 .pt 张量文件并保存在输出目录中，使用 DAC 模型。
    实现断点续传功能，并根据WAV文件数量实现等待逻辑。
    """
    print("--- WAV 转 PT 转换开始 ---")
    print(f"输入 WAV 目录: {wav_input_dir}")
    print(f"输出 .pt 目录: {pt_output_dir}")
    print(f"DAC 模型路径: {model_path}")
    print(f"成功后删除源 WAV: {delete_source_wav_on_success}")
    print(f"扫描JSON间隔: {WAIT_INTERVAL_SECONDS}s")

    if not os.path.isdir(wav_input_dir):
        print(f"错误: 输入 WAV 目录 '{wav_input_dir}' 未找到。脚本将退出。")
        return

    try:
        os.makedirs(pt_output_dir, exist_ok=True)
    except OSError as e:
        print(f"错误: 无法创建输出 .pt 目录 '{pt_output_dir}': {e}。脚本将退出。")
        return

    # --- 为中断的运行执行预清理步骤 (运行一次) ---
    print("\n--- 正在为先前转换的文件执行预清理 ---")
    pre_deleted_wav_count = 0
    json_updated_in_pre_cleanup = False # Flag to save JSON only if changes were made
    
    status_data_for_pre_cleanup = load_status_from_json(STATUS_JSON_PATH)

    if os.path.isdir(pt_output_dir) and os.path.isdir(wav_input_dir) : # Ensure both dirs exist
        try:
            pt_files = os.listdir(pt_output_dir)
            for pt_filename in pt_files:
                if pt_filename.lower().endswith(".pt"):
                    base, _ = os.path.splitext(pt_filename)
                    potential_source_wav_filename = f"{base}.wav"
                    corresponding_source_path = os.path.join(wav_input_dir, potential_source_wav_filename)
                    
                    # 首先尝试删除源WAV（如果设置了）
                    if delete_source_wav_on_success and os.path.exists(corresponding_source_path):
                        try:
                            os.remove(corresponding_source_path)
                            print(f"  预清理: 已删除源文件 '{corresponding_source_path}' (因存在对应的 .pt 文件 '{pt_filename}' 且设置了删除选项)")
                            pre_deleted_wav_count += 1
                        except OSError as e_remove:
                            print(f"  预清理错误: 删除源文件 '{corresponding_source_path}' 失败: {e_remove}")

                    # 更新JSON状态以反映PT文件的存在
                    current_status_entry = status_data_for_pre_cleanup.get(potential_source_wav_filename)
                    
                    if current_status_entry is not None:
                        # 条目存在，检查并更新第一个标记
                        if isinstance(current_status_entry, list) and len(current_status_entry) == 2:
                            if current_status_entry[0] != 1:
                                print(f"    预清理: JSON中 '{potential_source_wav_filename}' 的状态从 {current_status_entry} 更新为 [1, {current_status_entry[1]}]。")
                                status_data_for_pre_cleanup[potential_source_wav_filename][0] = 1
                                json_updated_in_pre_cleanup = True
                            # else:
                            #     print(f"    预清理: JSON中 '{potential_source_wav_filename}' 的状态已为 [1, x]。无需更改第一个标记。")
                        else:
                            # 格式不正确，覆盖它
                            print(f"    预清理: JSON中 '{potential_source_wav_filename}' 的状态格式不正确 ({current_status_entry})。已修正为 [1, 0]。")
                            status_data_for_pre_cleanup[potential_source_wav_filename] = [1, 0] # 假设第二个阶段未完成
                            json_updated_in_pre_cleanup = True
                    else:
                        # 条目不存在，添加它
                        print(f"    预清理: '{potential_source_wav_filename}' (对应PT '{pt_filename}') 不在JSON中。已添加并标记状态为 [1, 0]。")
                        status_data_for_pre_cleanup[potential_source_wav_filename] = [1, 0]
                        json_updated_in_pre_cleanup = True
                                
        except OSError as e_list_dirs:
            print(f"  预清理错误: 无法读取 .pt 输出目录 '{pt_output_dir}' 或WAV输入目录 '{wav_input_dir}': {e_list_dirs}")
    
    if json_updated_in_pre_cleanup:
        save_status_to_json(STATUS_JSON_PATH, status_data_for_pre_cleanup)
        print(f"预清理期间JSON状态已更新并保存。")
        
    print(f"预清理完成。已删除 {pre_deleted_wav_count} 个孤立的源文件。")
    # --- 预清理结束 ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n将使用设备: {device}")

    model = None
    try:
        print(f"正在从 '{model_path}' 加载 DAC 模型...")
        model = load_dac_model_from_safetensors(model_path, device)
        if model is None:
            print("DAC 模型加载失败。脚本将退出处理循环。") # 从"脚本退出"更改而来
            return # 退出函数，主循环不会开始
        print("DAC 模型加载成功。")
    except FileNotFoundError as e_load_model:
        print(f"模型加载错误: {e_load_model}。请确保模型路径和文件正确。脚本将退出处理循环。")
        return
    except Exception as e_load_model_generic:
        print(f"加载 DAC 模型 '{model_path}' 时发生未知错误: {e_load_model_generic}。脚本将退出处理循环。")
        return

    # --- 主处理循环 (JSON驱动) ---
    print(f"--- 开始JSON驱动的WAV到PT转换循环 (状态文件: {STATUS_JSON_PATH}) ---")
    while True:
        status_data = load_status_from_json(STATUS_JSON_PATH)
        
        wav_filenames_to_process = [
            fname for fname, status_val in status_data.items() 
            if isinstance(status_val, list) and len(status_val) == 2 and status_val[0] == 0
        ]

        if not wav_filenames_to_process:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: 在JSON中未找到标记为0的WAV文件。等待 {WAIT_INTERVAL_SECONDS} 秒...")
            time.sleep(WAIT_INTERVAL_SECONDS)
            continue # 重新扫描JSON

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: 从JSON找到 {len(wav_filenames_to_process)} 个待处理的WAV文件 (标记为0)。")
        
        converted_in_this_run = 0
        failed_in_this_run = 0
        skipped_missing_in_this_run = 0

        for wav_filename in wav_filenames_to_process: # 迭代JSON中状态为0的名称
            wav_filepath = os.path.join(wav_input_dir, wav_filename)
            
            # _process_single_wav_to_pt 处理检查已存在的 .pt 文件
            success = _process_single_wav_to_pt(wav_filepath, pt_output_dir, model, device)
            
            current_status_data = load_status_from_json(STATUS_JSON_PATH) # 为每个文件的原子性操作重新加载

            if success:
                # 如果 _process_single_wav_to_pt 因为 .pt 文件已存在而返回True， 
                # 或者因为它成功转换了。
                original_second_status = current_status_data.get(wav_filename, [0,0])[1] if isinstance(current_status_data.get(wav_filename), list) and len(current_status_data.get(wav_filename)) == 2 else 0
                current_status_data[wav_filename] = [1, original_second_status] # 标记第一个阶段完成
                save_status_to_json(STATUS_JSON_PATH, current_status_data)
                print(f"  JSON状态已为 '{wav_filename}' 更新为 [1, {original_second_status}]。")
                converted_in_this_run += 1

                if delete_source_wav_on_success and os.path.exists(wav_filepath):
                    # 检查成功是否因为新的转换，而不仅仅是为删除安全而存在的预先存在的pt文件
                    # 如果由于已存在的 .pt 文件而跳过，_process_single_wav_to_pt 会打印信息
                    # 如果需要，我们可以增强它以返回更具体的状态。
                    # 目前，如果 success=True 且 wav 文件存在，则尝试删除。
                    pt_base_name = os.path.splitext(wav_filename)[0] + ".pt"
                    pt_file_path_check = os.path.join(pt_output_dir, pt_base_name)
                    if os.path.exists(pt_file_path_check): # 在删除WAV之前确保PT文件存在
                        try:
                            os.remove(wav_filepath)
                            print(f"  已删除源WAV文件: {wav_filepath}")
                        except OSError as e_remove:
                            print(f"  删除源WAV文件 '{wav_filepath}' 失败: {e_remove}")
                    else:
                        print(f"  警告: PT文件 '{pt_file_path_check}' 未找到，即使转换报告成功。不删除源WAV '{wav_filepath}'.")
            else:
                # 此文件转换失败（例如，_process_single_wav_to_pt 未找到文件，或处理错误）
                failed_in_this_run += 1
                original_second_status = current_status_data.get(wav_filename, [0,0])[1] if isinstance(current_status_data.get(wav_filename), list) and len(current_status_data.get(wav_filename)) == 2 else 0
                
                if not os.path.exists(wav_filepath):
                    print(f"  文件 '{wav_filepath}' (来自JSON) 在文件系统中未找到。标记为错误状态([2, {original_second_status}])以避免重试。")
                    current_status_data[wav_filename] = [2, original_second_status] # 2 表示文件未找到/错误
                    skipped_missing_in_this_run +=1
                else:
                    print(f"  转换 '{wav_filename}' 失败。将在JSON中标记为错误状态([2, {original_second_status}])以避免重试。")
                    current_status_data[wav_filename] = [2, original_second_status] # 标记为错误以避免重试
                
                save_status_to_json(STATUS_JSON_PATH, current_status_data)
        
        print(f"本轮处理总结: 成功/已存在PT {converted_in_this_run}, 转换失败/标记错误 {failed_in_this_run}, 源文件失踪 {skipped_missing_in_this_run}")

        # 如果尝试了工作，则此处没有显式休眠，循环将在短时间处理后立即重新评估JSON。
        # 仅当 wav_filenames_to_process 最初为空时才等待。
        if not wav_filenames_to_process and converted_in_this_run == 0 and failed_in_this_run == 0:
            # 这种情况由顶部的检查处理，但作为循环逻辑的安全措施。
            pass 

    # --- 如果有东西中断或手动停止，循环结束 --- 
    print("--- WAV 转 PT JSON驱动的转换循环已停止 --- ")

if __name__ == "__main__":
    print("开始执行 WAV 到 PT 转换脚本 (JSON驱动)...")
    print(f"将使用状态JSON文件: {STATUS_JSON_PATH}")
    process_wav_to_pt_with_resume() 