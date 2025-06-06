import os
import json
import random
import shutil
import sys
import time
import multiprocessing

try:
    import pretty_midi
except ImportError:
    print("错误：pretty_midi 库未找到。请通过 'pip install pretty_midi' 安装它。", file=sys.stderr)
    sys.exit(1)

def process_single_xyz_midi_directory(args_tuple):
    """
    工作函数，由进程池中的每个进程执行。
    处理一个指定的 .../X/Y/Z/ 目录，找到其中每个音轨ID文件夹下的最长MIDI，
    并匹配对应的JSON文件。会检查 stop_event 来提前退出。
    """
    xyz_midi_dir_path, path_parts_xyz, json_source_root_dir, midi_extension, stop_event = args_tuple
    local_candidate_pairs = []
    subdir1, subdir2, subdir3 = path_parts_xyz 

    if not os.path.isdir(xyz_midi_dir_path):
        return local_candidate_pairs

    for track_id_folder_name in os.listdir(xyz_midi_dir_path):
        if stop_event.is_set(): # 检查是否需要提前停止
            # print(f"  工作进程 {os.getpid()} 在 {xyz_midi_dir_path} 中检测到停止信号，已处理部分 track_id。")
            break # 停止处理当前 X/Y/Z 目录下的更多 track_id 文件夹

        potential_track_id = track_id_folder_name
        if not (potential_track_id.startswith("TR") and len(potential_track_id) == 18):
            continue

        current_track_id_midi_folder = os.path.join(xyz_midi_dir_path, track_id_folder_name)
        if not os.path.isdir(current_track_id_midi_folder):
            continue

        longest_midi_path_for_track = None
        max_duration = -1.0
        try:
            for item_in_subdir in os.listdir(current_track_id_midi_folder):
                if item_in_subdir.lower().endswith(midi_extension):
                    current_midi_file_path = os.path.join(current_track_id_midi_folder, item_in_subdir)
                    try:
                        midi_data = pretty_midi.PrettyMIDI(current_midi_file_path)
                        duration = midi_data.get_end_time()
                        if duration > max_duration:
                            max_duration = duration
                            longest_midi_path_for_track = current_midi_file_path
                    except Exception: 
                        pass 
        except OSError: 
            continue
        
        if not longest_midi_path_for_track:
            continue

        json_filename = potential_track_id + ".json"
        json_file_path = os.path.join(json_source_root_dir, subdir1, subdir2, subdir3, json_filename)

        if not os.path.exists(json_file_path):
            continue
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f_json:
                json_content = json.load(f_json)
            
            tags = json_content.get("tags", [])
            if not (tags and isinstance(tags, list) and len(tags) > 0):
                continue

            similars_dict = {s[0]: s[1] for s in json_content.get("similars", []) if isinstance(s, list) and len(s) == 2}
            
            local_candidate_pairs.append({
                "midi_path": longest_midi_path_for_track,
                "json_path": json_file_path,
                "track_id": potential_track_id,
                "tags_count": len(tags),
                "similars_dict": similars_dict
            })
        except json.JSONDecodeError:
            pass
        except Exception:
            pass
            
    return local_candidate_pairs


def select_and_copy_matched_pairs(
    midi_source_root_dir,
    json_source_root_dir,
    json_target_dir,
    midi_target_dir,
    count=1000,
    similarity_threshold=0.5,
    midi_extension=".mid",
    num_processes=None,
    early_candidate_threshold=2000 # 新增参数：提前停止收集的候选对数量阈值
):
    """
    从MIDI文件开始，并行扫描X/Y/Z目录，选择时长最长的MIDI及其对应的、有标签的JSON文件，
    同时满足JSON间的相似度约束，并优先选择标签数量多的JSON。
    当收集到的原始候选对达到 early_candidate_threshold 时，会尝试停止进一步的收集。
    确保最终在目标目录中JSON和MIDI文件一一对应。
    """
    all_candidate_pairs_info = []
    tasks_for_workers = []

    # 创建一个停止事件，用于通知工作进程停止
    # 对于需要跨进程传递的Event，通常使用Manager().Event()
    # 但对于Pool的map/imap，简单的multiprocessing.Event()通常也可以（因为它会被pickle）
    manager = multiprocessing.Manager()
    stop_event = manager.Event()

    print(f"准备并行扫描任务，根目录: {midi_source_root_dir}")
    
    if not os.path.isdir(midi_source_root_dir):
        print(f"错误: MIDI源根目录 '{midi_source_root_dir}' 不存在或不是一个目录。")
        return []

    level_0_dirs = [d for d in os.listdir(midi_source_root_dir) if os.path.isdir(os.path.join(midi_source_root_dir, d))]
    for l0_dir in level_0_dirs: # X
        if stop_event.is_set(): break # 如果已经设置停止，则不再准备新任务
        l0_path = os.path.join(midi_source_root_dir, l0_dir)
        level_1_dirs = [d for d in os.listdir(l0_path) if os.path.isdir(os.path.join(l0_path, d))]
        for l1_dir in level_1_dirs: # Y
            if stop_event.is_set(): break
            l1_path = os.path.join(l0_path, l1_dir)
            level_2_dirs = [d for d in os.listdir(l1_path) if os.path.isdir(os.path.join(l1_path, d))]
            for l2_dir in level_2_dirs: # Z
                if stop_event.is_set(): break
                xyz_full_midi_path = os.path.join(l1_path, l2_dir)
                path_parts_xyz_tuple = (l0_dir, l1_dir, l2_dir) 
                tasks_for_workers.append(
                    (xyz_full_midi_path, path_parts_xyz_tuple, json_source_root_dir, midi_extension, stop_event)
                )
            if stop_event.is_set(): break
        if stop_event.is_set(): break

    if not tasks_for_workers and not stop_event.is_set(): # 确保不是因为提前停止而没有任务
        print("错误：未能从MIDI源目录中识别出任何有效的X/Y/Z扫描任务。请检查路径和结构。")
        return []
    elif not tasks_for_workers and stop_event.is_set():
        print("信息：由于已达到候选对收集上限，未准备更多扫描任务。")

    if tasks_for_workers: # 只有在有任务需要处理时才启动进程池
        print(f"共识别出 {len(tasks_for_workers)} 个X/Y/Z目录需要并行处理。")
        
        if num_processes is None:
            num_cpus = os.cpu_count()
            num_processes = num_cpus if num_cpus is not None else 1
        num_processes = max(1, num_processes) 
                
        print(f"使用 {num_processes} 个进程进行处理...")

        processed_tasks_count = 0
        start_pool_time = time.time()

        with multiprocessing.Pool(processes=num_processes) as pool:
            results_iterator = pool.imap_unordered(process_single_xyz_midi_directory, tasks_for_workers)
            
            print_interval = max(1, len(tasks_for_workers) // 20) 

            for single_xyz_results in results_iterator:
                all_candidate_pairs_info.extend(single_xyz_results)
                processed_tasks_count += 1
                if processed_tasks_count % print_interval == 0 or processed_tasks_count == len(tasks_for_workers):
                    elapsed_time = time.time() - start_pool_time
                    print(f"  已处理 {processed_tasks_count}/{len(tasks_for_workers)} 个X/Y/Z目录... 候选对总数: {len(all_candidate_pairs_info)}. 用时: {elapsed_time:.2f}s")
                
                # 检查是否达到提前停止收集的阈值
                if early_candidate_threshold is not None and len(all_candidate_pairs_info) >= early_candidate_threshold:
                    if not stop_event.is_set(): # 仅在第一次达到时设置并打印
                        print(f"  已收集到 {len(all_candidate_pairs_info)} 个候选对，达到阈值 {early_candidate_threshold}。正在设置停止信号...")
                        stop_event.set() 
                    # 注意：设置停止信号后，我们仍然需要处理完当前已提交到imap_unordered并正在运行或已完成的任务的结果
                    # imap_unordered 会继续迭代直到所有提交的任务都返回结果（即使是空结果）
                    # 如果想更强制地停止，需要在设置stop_event后考虑关闭或终止pool，但这会使结果收集复杂化
                    # 当前设计是合作式停止，工作进程会检查信号并在处理下一个主要单元（track_id_folder）前退出。

        end_pool_time = time.time()
        print(f"所有已提交的X/Y/Z目录扫描任务完成，用时: {end_pool_time - start_pool_time:.2f} 秒。")
    
    print(f"总共找到 {len(all_candidate_pairs_info)} 个候选 (MIDI, JSON带标签) 对。")

    if not all_candidate_pairs_info:
        print("未找到有效的 (最长MIDI, 对应JSON带标签) 对。")
        return []

    # 后续的排序、相似度筛选和复制逻辑与之前相同
    print(f"将从 {len(all_candidate_pairs_info)} 个候选对中进行排序和筛选...")
    if len(all_candidate_pairs_info) > count * 1.5: # 如果收集到的候选远超所需，先随机打乱以避免极端情况
        random.shuffle(all_candidate_pairs_info)
    all_candidate_pairs_info.sort(key=lambda x: x["tags_count"], reverse=True)

    selected_pairs_final = [] 
    print(f"正在从排序后的候选对中选择最多 {count} 对 (相似度阈值: {similarity_threshold})...")
    for candidate_info in all_candidate_pairs_info:
        if len(selected_pairs_final) >= count:
            break
        is_compatible = True
        candidate_track_id = candidate_info["track_id"]
        candidate_similars = candidate_info["similars_dict"]
        for already_selected_info in selected_pairs_final:
            selected_track_id_in_list = already_selected_info["track_id"]
            selected_similars_in_list = already_selected_info["similars_dict"]
            if candidate_track_id in selected_similars_in_list and selected_similars_in_list[candidate_track_id] > similarity_threshold:
                is_compatible = False
                break
            if selected_track_id_in_list in candidate_similars and candidate_similars[selected_track_id_in_list] > similarity_threshold:
                is_compatible = False
                break
        if is_compatible:
            selected_pairs_final.append(candidate_info)
    
    if not selected_pairs_final:
        print("在相似度筛选后未能选择任何文件对。")
        return []
    if len(selected_pairs_final) < count:
        print(f"警告: 由于约束，只选择了 {len(selected_pairs_final)} 对 (目标 {count} 对)。")

    os.makedirs(json_target_dir, exist_ok=True)
    os.makedirs(midi_target_dir, exist_ok=True)
    final_copied_track_ids = []
    num_json_copied = 0
    num_midi_copied = 0

    print(f"正在复制 {len(selected_pairs_final)} 个选定的JSON和MIDI文件对...")
    for pair_to_copy in selected_pairs_final:
        json_src_path = pair_to_copy["json_path"]
        midi_src_path = pair_to_copy["midi_path"]
        track_id = pair_to_copy["track_id"] 
        source_json_actual_filename = os.path.basename(json_src_path)
        json_dest_filename = source_json_actual_filename 
        json_base_name = os.path.splitext(source_json_actual_filename)[0]
        midi_dest_filename = json_base_name + midi_extension 
        json_dest_path = os.path.join(json_target_dir, json_dest_filename)
        midi_dest_path = os.path.join(midi_target_dir, midi_dest_filename)
        try:
            shutil.copy2(json_src_path, json_dest_path)
            num_json_copied += 1
            try:
                shutil.copy2(midi_src_path, midi_dest_path)
                num_midi_copied += 1
                final_copied_track_ids.append(track_id) 
            except Exception as e_midi_copy:
                print(f"错误: 复制MIDI文件 {midi_src_path} 到 {midi_dest_path} 失败: {e_midi_copy}", file=sys.stderr)
                if os.path.exists(json_dest_path):
                    try:
                        os.remove(json_dest_path)
                        num_json_copied -= 1
                        print(f"已移除JSON文件 {json_dest_path} 因为其对应的MIDI复制失败。")
                    except OSError as e_remove:
                        print(f"错误: 移除JSON文件 {json_dest_path} 失败: {e_remove}", file=sys.stderr)
        except Exception as e_json_copy:
            print(f"错误: 复制JSON文件 {json_src_path} 到 {json_dest_path} 失败: {e_json_copy}", file=sys.stderr)
            
    print(f"处理完成。成功复制 {num_json_copied} 个JSON文件和 {num_midi_copied} 个MIDI文件。")
    if num_json_copied != num_midi_copied:
        print(f"警告: 复制的JSON ({num_json_copied}) 和MIDI ({num_midi_copied}) 文件数量不匹配！", file=sys.stderr)
    elif num_json_copied < len(selected_pairs_final): 
        print(f"警告: 并非所有选定的文件对 ({len(selected_pairs_final)}) 都成功复制。实际JSON复制数: {num_json_copied}", file=sys.stderr)

    return final_copied_track_ids


if __name__ == "__main__":
    multiprocessing.freeze_support() 
    # --- 配置 ---
    lastfm_train_json_source_dir = r"F:\music\lastfm_train"
    lmd_matched_midi_source_dir = r"F:\music\lmd_matched.tar\lmd_matched" 

    output_selected_json_dir = r"F:\music\selected_json_files"
    output_selected_midi_dir = r"F:\music\selected_midi_files"

    num_pairs_to_select = 3000
    similarity_constraint_threshold = 0.5
    source_midi_file_extension = ".mid" 
    number_of_processes = 4 # 修改：用户要求使用4个进程
    early_stop_candidate_count = 5000 # 当原始候选对达到此数量时，尝试停止收集新候选
    # --- 配置结束 ---

    print("开始处理音乐数据对...")
    start_total_time = time.time()

    successfully_processed_track_ids = select_and_copy_matched_pairs(
        midi_source_root_dir=lmd_matched_midi_source_dir,
        json_source_root_dir=lastfm_train_json_source_dir,
        json_target_dir=output_selected_json_dir,
        midi_target_dir=output_selected_midi_dir,
        count=num_pairs_to_select,
        similarity_threshold=similarity_constraint_threshold,
        midi_extension=source_midi_file_extension,
        num_processes=number_of_processes,
        early_candidate_threshold=early_stop_candidate_count
    )

    end_total_time = time.time()
    print(f"音乐数据处理完成。总用时: {end_total_time - start_total_time:.2f} 秒。")

    if successfully_processed_track_ids:
        print(f"成功处理并选定 {len(successfully_processed_track_ids)} 个音轨对。")
        if len(successfully_processed_track_ids) < num_pairs_to_select:
            print(f"注意: 选定的音轨对数量 ({len(successfully_processed_track_ids)}) 少于目标数量 ({num_pairs_to_select}).")
    else:
        print("未能选择或处理任何音轨对。")

    print("音乐数据处理完成。") 