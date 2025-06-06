---
license: mit
tags:
- DAC
- Descript Audio Codec
- PyTorch
---

# Descript Audio Codec (DAC)
DAC is the state-of-the-art audio tokenizer with improvement upon the previous tokenizers like SoundStream and EnCodec.

This model card provides an easy-to-use API for a *pretrained DAC* [1] for 44.1khz audio whose backbone and pretrained weights are from [its original reposotiry](https://github.com/descriptinc/descript-audio-codec). With this API, you can encode and decode by a single line of code either using CPU or GPU. Furhtermore, it supports chunk-based processing for memory-efficient processing, especially important for GPU processing. 






### Model variations
There are three types of model depending on an input audio sampling rate. 

| Model | Input audio sampling rate [khz] |
| ------------------ | ----------------- |
| [`hance-ai/descript-audio-codec-44khz`](https://huggingface.co/hance-ai/descript-audio-codec-44khz) | 44.1khz |
| [`hance-ai/descript-audio-codec-24khz`](https://huggingface.co/hance-ai/descript-audio-codec-24khz) | 24khz |
| [`hance-ai/descript-audio-codec-16khz`](https://huggingface.co/hance-ai/descript-audio-codec-16khz) | 16khz |




# Dependency
See `requirements.txt`.





# Usage

### Load
```python
from transformers import AutoModel

# device setting
device = 'cpu'  # or 'cuda:0'

# load
model = AutoModel.from_pretrained('hance-ai/descript-audio-codec-44khz', trust_remote_code=True)
model.to(device)
```

### Encode
```python
audio_filename = 'path/example_audio.wav'
zq, s = model.encode(audio_filename)
```
`zq` is discrete embeddings with dimension of (1, num_RVQ_codebooks, token_length) and `s` is a token sequence with dimension of (1, num_RVQ_codebooks, token_length). 


### Decode
```python
# decoding from `zq`
waveform = model.decode(zq=zq)  # (1, 1, audio_length); the output has a mono channel.

# decoding from `s`
waveform = model.decode(s=s)  # (1, 1, audio_length); the output has a mono channel.
```

### Save a waveform as an audio file
```python
model.waveform_to_audiofile(waveform, 'out.wav')
```

### Save and load tokens
```python
model.save_tensor(s, 'tokens.pt')
loaded_s = model.load_tensor('tokens.pt')
```








# Runtime

To give you a brief idea, the following table reports average runtime on CPU and GPU to encode and decode 10s audio. The runtime is measured in second. The used CPU is Intel Core i9 11900K and GPU is RTX3060.
```
|           Task  |   CPU   |   GPU   |
|-----------------|---------|---------|
|    Encoding     |   6.71  |  0.19   |
|    Decoding     |   15.4  |  0.31   |
```
The decoding process takes a longer simply because the decoder is larger than the encoder. 









# Technical Discussion

### Chunk-based Processing
It's introduced for memory-efficient processing for both encoding and decoding. 
For encoding, we simply chunk an audio into N chunks and process them iteratively.
Similarly, for decoding, we chunk a token set into M chunks of token subsets and process them iteratively.
However, the decoding process with naive chunking causes an artifact in the decoded audio. 
That is because the decoder reconstructs audio given multiple neighboring tokens (i.e., multiple neighboring tokens for a segment of audio) rather than a single token for a segment of audio. 

To tackle the problem, we introduce overlap between the chunks in the decoding, parameterized by `decoding_overlap_rate` in the model. By default, we introduce 10% of overlap between the chunks. Then, two subsequent chunks reuslt in two segments of audio with 10% overlap, and the overlap is averaged out for smoothing.

The following figure compares reconstructed audio with and without the overlapping.
<p align="center">
<img src=".fig/artifact-dac-decoding without overlap.png" alt="" width=50%>
</p>







# References
[1] Kumar, Rithesh, et al. "High-fidelity audio compression with improved rvqgan." Advances in Neural Information Processing Systems 36 (2024).



<!-- contributions 
- chunk processing 
- add device parameter in the test notebook
-->
