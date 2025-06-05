# JambaMusicModel
使用上下文长度极长同时增加部分注意力机制的Jamba模型和decript-audio-codec编解码器，用于LLM驱动的音频生成

## 描述

这是一个模型的创建与训练的项目，训练的模型用于输入经过训练的LLM模型生成的中间值生成DAC编解码器（decript-audio-codec编解码器）的s向量并能够通过DAC编解码器转换成音频文件，中间值具体是在JambaForAudioGeneration_S_In_S_Out的StyleEncoder卷积头生成的风格向量以及大约100个时间步长（可以自己设置）的。注：本项目使用linux系统。

## 功能特点

*   基于 Jamba 架构的 `JambaForAudioGeneration_S_In_S_Out` 模型。
*   支持从 DAC 编码的 's' 张量进行音频到音频的转换。
*   实现了详细的训练、验证和检查点管理逻辑。
*   支持计划采样和混合精度训练。
*   包含可配置的 Style Encoder。

## 安装

1.  克隆仓库 (如果您已经克隆了，可以跳过此步):
    ```bash
    git clone https://github.com/raw-poplar/JambaMusicModel
    cd JambaMusicModel
    ```
2.  (推荐) 创建并激活一个 Python 虚拟环境:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  安装依赖:
    ```bash
    pip install -r requirements.txt
    ```

## 数据准备

训练数据应为 `.pt` 文件，每个文件包含一个至少名为 `'s'` 的 PyTorch 张量。该张量是来自 `descript-audio-codec` (DAC) 的编码表示。

*   **数据格式**: 's' 张量期望的原始形状为 `(1, Q, T_orig)` 或 `(Q, T_orig)` (其中 Q 是量化器数量，T_orig 是原始时间步数)，脚本内部会将其处理为 `(T_orig, Q)`。
*   **配置路径**: 在 `train_jamba_s_in_s_out_step.py` 脚本中，确保 `DAC_PT_DIR` 变量指向包含这些 `.pt` 文件的目录。默认设置为 `/root/autodl-tmp/modelTrain/jambaDataset2`，您需要根据您的实际路径进行修改。

## 使用

1.  **配置训练参数**: 打开 `train_jamba_s_in_s_out_step.py` 文件，根据您的需求修改顶部的配置参数，例如：
    *   `DAC_PT_DIR`: 您的数据集路径。
    *   `PRETRAINED_MODEL_LOAD_PATH`: 如果有预训练模型，请指定路径。
    *   `STEP_MODEL_SAVE_DIR`: 模型保存路径。
    *   `TOTAL_TRAINING_EPOCHS`, `EPOCHS_FILES_NUM`, `LEARNING_RATE` 等训练超参数。
    *   `JAMBA_S_CONFIG_PARAMS`: Jamba 模型的具体配置参数，如 `d_model`, `n_layer`, `dac_codebook_size`, `dac_num_quantizers` 等。**特别注意 `dac_codebook_size` 和 `dac_num_quantizers` 必须与您 DAC 数据的设置一致。**

2.  **开始训练**:
    ```bash
    python train_jamba_s_in_s_out_step.py
    ```
    训练日志将输出到控制台，并保存在 `STEP_MODEL_SAVE_DIR` 下的 `training_log_s_in_s_out_step.log` 文件中。
    模型检查点会保存在 `STEP_MODEL_SAVE_DIR` 下的 `checkpoint_file_<count>_<filename>` 子目录中。
    最佳模型会保存在 `STEP_MODEL_SAVE_DIR` 下的 `best_model_s_in_s_out` 子目录中。

## 模型架构

本项目使用 `JambaForAudioGeneration_S_In_S_Out` 模型，其配置由 `JambaSConfig` 定义。核心组件包括：
*   Jamba/Mamba 块 (结合了 Mamba 层和可选的注意力层)。
*   Style Encoder 用于提取音频风格信息。
*   详细信息请参考 `jamba_audio_s_in_s_out.py` (如果这是您的模型定义文件) 和 `train_jamba_s_in_s_out_step.py` 中的 `JAMBA_S_CONFIG_PARAMS`。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。请查阅 `LICENSE` 文件获取详细信息。

## 注意事项

*   确保您的环境中已正确安装 PyTorch 并能访问 GPU (如果配置为使用 CUDA)。
*   `descript-audio-codec` 的安装和使用可能需要额外的依赖或设置，请参考其官方文档。

## 致谢 (Acknowledgments)

本项目在开发过程中使用了以下优秀的开源项目和库，特此感谢：

*   **Descript Audio Codec**: 由 Descript AI 研究团队开发的音频编解码器。
    *   GitHub: [https://github.com/descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)
*   **Mamba2 实现**: 本项目使用了 [https://github.com/tommyip/mamba2-minimal] 上提供的 Mamba2 模型的一个实现。感谢其作者 [tommyip] 的工作。

<!-- 如果您还参考了其他论文、代码库或受到了某些工作的启发，也请在这里列出。 --> 
