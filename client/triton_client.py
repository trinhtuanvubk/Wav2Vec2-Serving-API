import torch
import tritonhttpclient

import librosa
import numpy as np
# from transformers import Wav2Vec2Processor


input_name = 'audio_input'
output_name = 'transcript_output'
VERBOSE = False
model_name = 'ensemble'
url = '0.0.0.0:8050'
model_version = '1'

# processor = Wav2Vec2Processor.from_pretrained("./processor/")



triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)   

data, sr = librosa.load("../sample/haimy_voice.wav", sr=16000)
# features = processor(data, sampling_rate=sr, return_tensors="pt")
# input_values = features.input_values
input_values=np.expand_dims(data, axis=0)
print(input_values.shape)
input = tritonhttpclient.InferInput(input_name, input_values.shape, 'FP32')
input.set_data_from_numpy(input_values)
output = tritonhttpclient.InferRequestedOutput(output_name)    


response_triton = triton_client.infer(model_name, model_version=model_version, inputs=[input], outputs=[output])
raw_transcript = response_triton.as_numpy('transcript_output')
final_transcript = [i.decode("utf-8") for i in raw_transcript]

print(final_transcript)