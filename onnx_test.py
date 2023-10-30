import torch #must first
import onnxruntime
import librosa
import numpy as np
from transformers import Wav2Vec2Processor

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)

print(onnxruntime.get_device())

processor = Wav2Vec2Processor.from_pretrained("./models/w2v2-base-250h/processor")
data, sr = librosa.load("./sample/haimy_voice.wav", sr=16000)

sess = onnxruntime.InferenceSession('./models/w2v2-base-250h/model/model.onnx',providers=['CUDAExecutionProvider'] )
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
outputs = [x.name for x in sess.get_outputs()]

y = sess.run(outputs, {sess.get_inputs()[0].name: np.expand_dims(data, axis=0)})[0]

print(y.shape)

logit = np.argmax(y, axis=-1)
print(logit.shape)
output = processor.batch_decode(logit)
print(output[0])