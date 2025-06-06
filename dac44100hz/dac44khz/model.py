from typing import Union

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torchaudio.transforms as transforms
from transformers import PretrainedConfig, PreTrainedModel

import dac
# from audiotools import AudioSignal

from dac import utils
from dac import model


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


class DACConfig(PretrainedConfig):
    model_type = 'dac'

    def __init__(self, 
                 model_type_by_sampling_freq:str='44khz',
                 encoding_chunk_size_in_sec:int=1,
                 decoding_chunk_rate:float=0.1,
                 decoding_overlap_rate:float=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        """
        Initializes the model object.
        Args:
            model_type_by_sampling_freq (str, optional): The model type based on the sampling frequency. Defaults to '44khz'. Choose among ['44khz', '24khz', '16khz']
            encoding_chunk_size_in_sec (int, optional): The size of the encoding chunk in seconds. Defaults to 1.
            decoding_chunk_rate (float, optional): The decoding chunk rate. Must be between 0 and 1. Defaults to 0.1.
            decoding_overlap_rate (float, optional): The decoding overlap rate. Must be between 0 and 1. Defaults to 0.1.
            **kwargs: Additional keyword arguments.
        Raises:
            AssertionError: If the model_type_by_sampling_freq is not one of ['44khz', '24khz', '16khz'].
            AssertionError: If the decoding_chunk_rate is not between 0 and 1.
            AssertionError: If the decoding_overlap_rate is not between 0 and 1.
        """
        self.model_type_by_sampling_freq = model_type_by_sampling_freq
        self.encoding_chunk_size_in_sec = encoding_chunk_size_in_sec
        self.decoding_chunk_rate = decoding_chunk_rate
        self.decoding_overlap_rate = decoding_overlap_rate

        assert model_type_by_sampling_freq.lower() in ['44khz', '24khz', '16khz']
        assert decoding_chunk_rate > 0 and decoding_chunk_rate <= 1.0, '`decoding_chunk_rate` must be bewteen 0 and 1.'
        assert decoding_overlap_rate >= 0 and decoding_overlap_rate < 1.0, '`decoding_overlap_rate` must be bewteen 0 and 1.'



class DAC(PreTrainedModel):
    config_class = DACConfig

    def __init__(self, config):
        super().__init__(config)

        self.model_type_by_sampling_freq = config.model_type_by_sampling_freq.lower()
        self.model_type_by_sampling_freq_int = {'44khz':44100, '24khz':24000, '16khz':16000}[self.model_type_by_sampling_freq]
        self.encoding_chunk_size_in_sec = config.encoding_chunk_size_in_sec
        self.decoding_chunk_rate = config.decoding_chunk_rate
        self.decoding_overlap_rate = config.decoding_overlap_rate


        dac_path = dac.utils.download(model_type=self.model_type_by_sampling_freq)
        self.dac = dac.model.DAC.load(dac_path)
        self.dac.eval()
        freeze(self.dac)

        self.downsampling_rate = int(np.prod(self.dac.encoder_rates))  # 512
    
    def load_audio(self, filename:str):
        waveform, sample_rate = torchaudio.load(filename)  # waveform: (n_channels, length); sample_rate: const.
        return waveform, sample_rate
    
    def resample_audio(self, waveform:torch.FloatTensor, orig_sr:int, target_sr:int):
        """
        - sr: sampling rate
        - waveform: (n_channels, length)
        """
        if orig_sr == target_sr:
            return waveform

        converter = transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = converter(waveform)  # (n_channels, new_length)
        return waveform  # (n_channels, new_length)

    def to_mono_channel(self, waveform:torch.FloatTensor):
        """
        - waveform: (n_channels, length)
        """
        n_channels = waveform.shape[0]
        if n_channels > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # (1, length)
        return waveform  # (1, length)
    
    @torch.no_grad()
    def encode(self, audio_fname:str):
        self.eval()

        waveform, sr = self.load_audio(audio_fname)
        waveform = self.resample_audio(waveform, orig_sr=sr, target_sr=self.model_type_by_sampling_freq_int)
        sr = self.model_type_by_sampling_freq_int
        waveform = self.to_mono_channel(waveform)  # DAC accepts a mono channel only.
        
        zq, s = self._chunk_encoding(waveform, sr)
        return zq, s

    def _chunk_encoding(self, waveform:torch.FloatTensor, sr:int):
        # TODO: can I make it parallel?
        """
        waveform: (c l)
        """
        x = waveform  # brief varname
        x = x.unsqueeze(1)  # (b 1 l); add a null batch dim
        chunk_size = int(self.encoding_chunk_size_in_sec * sr)

        # adjust `chunk_size` to prevent any padding in `dac.preprocess`, which causes a gap between the mini-batches in the resulting music.
        remainer = chunk_size % self.dac.hop_length
        chunk_size = chunk_size-remainer

        # process
        zq_list, s_list = [], []
        audio_length = x.shape[-1]
        for start in range(0, audio_length, chunk_size):
            end = start + chunk_size
            chunk = x[:, :, start:end]
            chunk = self.dac.preprocess(chunk, sr)
            zq, s, _, _, _ = self.dac.encode(chunk.to(self.device))
            zq = zq.cpu()
            s = s.cpu()
            """
            "zq" : Tensor[B x D x T]
                Quantized continuous representation of input
                = summation of all the residual quantized vectors across every rvq level
                = E(x) = z = \sum_n^N{zq_n} where N is the number of codebooks
            "s" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
                *first element in the N dimension = first RVQ level
            """
            zq_list.append(zq)
            s_list.append(s)
            torch.cuda.empty_cache()
        
        zq = torch.cat(zq_list, dim=2).float()  # (1, d, length)
        s = torch.cat(s_list, dim=2).long()  # (1, n_rvq, length)

        return zq, s
    
    @torch.no_grad()
    def decode(self, *, zq:Union[torch.FloatTensor,None]=None, s:Union[torch.IntTensor,None]=None):
        """
        zq: (b, d, length)
        """
        if isinstance(zq,type(None)) and isinstance(s,type(None)):
            assert False, 'one of them must be valid.'
        self.eval()

        if not isinstance(zq,type(None)):
            waveform = self._chunk_decoding(zq)  # (b, 1, length); output always has a mono-channel.
        if not isinstance(s,type(None)):
            zq = self.code_to_zq(s)
            waveform = self._chunk_decoding(zq)  # (b, 1, length); output always has a mono-channel.

        return waveform
    
    def _chunk_decoding(self, zq:torch.FloatTensor):
        """
        zq: (b, d, length)
        """
        length = zq.shape[-1]
        chunk_size = round(int(self.decoding_chunk_rate * length))
        overlap_size = round(self.decoding_overlap_rate * chunk_size)  # overlap size in terms of token length
        overlap_size_in_data_space = round(overlap_size * self.downsampling_rate)
        waveform_concat = None
        for start in range(0, length, chunk_size-overlap_size):
            end = start + chunk_size
            chunk = zq[:,:, start:end]  # (b, d, chunk_size)
            waveform = self.dac.decode(chunk.to(self.device))  # (b, 1, chunk_size*self.downsampling_rate)
            waveform = waveform.cpu()

            waveform_len = waveform.shape[-1]
            if waveform_len < overlap_size_in_data_space:
                overlap_size_in_data_space = waveform_len
            
            if isinstance(waveform_concat, type(None)):
                waveform_concat = waveform.clone()
            else:
                if self.decoding_overlap_rate != 0.:
                    prev_x = waveform_concat[:,:,:-overlap_size_in_data_space]
                    rest_of_new_x = waveform[:,:,overlap_size_in_data_space:]
                    overlap_x_from_prev_x = waveform_concat[:,:,-overlap_size_in_data_space:]  # (b, 1, overlap_size_in_data_space)
                    overlap_x_from_new_x = waveform[:,:,:overlap_size_in_data_space]  # (b, 1, overlap_size_in_data_space)
                    if not overlap_x_from_new_x.shape[-1] == 0:
                        overlap = (overlap_x_from_prev_x + overlap_x_from_new_x) / 2  # take mean; maybe there's a better strategy but it seems to work fine.
                    else:
                        overlap = overlap_x_from_prev_x
                    waveform_concat = torch.cat((prev_x, overlap, rest_of_new_x), dim=-1)  # (b, 1, ..)
                else:
                    prev_x = waveform_concat
                    rest_of_new_x = waveform
                    waveform_concat = torch.cat((prev_x, rest_of_new_x), dim=-1)  # (b, 1, ..)
        return waveform_concat  # (b, 1, length)

    def code_to_zq(self, s:torch.IntTensor):
        """
        s: (b, n_rvq, length)
        """
        zq, _, _ = self.dac.quantizer.from_codes(s.to(self.device))  # zq: (b, d, length)
        zq = zq.cpu()
        return zq

    def save_tensor(self, tensor:torch.Tensor, fname:str) -> None:
        torch.save(tensor.cpu(), fname)
    
    def load_tensor(self, fname:str):
        return torch.load(fname)
    
    # def waveform_to_audiofile(self, waveform:torch.FloatTensor, fname:str) -> None:
    #     AudioSignal(waveform, sample_rate=self.model_type_by_sampling_freq_int).write(fname)

    def waveform_to_audiofile(self, waveform: torch.FloatTensor, fname: str) -> None:
        """
        将 PyTorch 张量格式的音频波形保存为文件。

        Args:
            waveform: 音频波形数据，PyTorch 张量，形状通常为 [通道数, 采样点数]。
            fname: 要保存的音频文件的路径和名称。
        """
        # 检查 waveform 的维度，torchaudio.save 需要 [channel, time]
        if waveform.ndim == 1:
            # 如果是单声道，添加一个通道维度
            waveform = waveform.unsqueeze(0)
        # NEW: Handle 3D tensor [batch, channel, time] if batch size is 1
        elif waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0) # Remove batch dimension
        elif waveform.ndim != 2: # Check if it's not 2D now
            # 如果维度仍然不符合要求 (不是 2D)，则抛出错误
            raise ValueError(f"Waveform tensor has unexpected dimensions after processing: {waveform.shape}. Expected [channel, time].")
        # 注意: 假设传入的 waveform 张量的形状已经是 [channel, time] 或 [time]

        # 使用 torchaudio.save 替代 AudioSignal().write()
        torchaudio.save(
            fname,  # 文件路径和名称 (位置参数)
            waveform,  # 波形张量 (应该是 [通道数, 采样点数])
            self.model_type_by_sampling_freq_int  # 采样率 (位置参数)
            # 可以添加其他 torchaudio.save 支持的参数，如 bits_per_sample, encoding 等
        )
