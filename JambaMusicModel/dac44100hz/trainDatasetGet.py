import os

import dacFunction
import torch

model_path = './dac44khz'
# windows 路径
input_path = 'F:\\music\\new_input'
output_path = 'D:\\music\\dataset\\dac_encoded_output_new'
# ubuntu路径
# input_path = '/root/autodl-tmp/modelTrain/wav'
# output_path = '/root/autodl-tmp/modelTrain/jambaModel'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = dacFunction.load_dac_model_from_safetensors(model_path, device)

# for foldname in os.listdir(input_path):
#     foldpath = os.path.join(input_path, foldname)
#     dacFunction.convert_audio_files_to_tensors(input_audio_dir=foldpath, output_tensor_dir=output_path, model=model)

dacFunction.convert_audio_files_to_tensors(
    input_audio_dir=input_path,
    output_tensor_dir=output_path,
    model=model
)