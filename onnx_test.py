import torch #must first
import json
import onnxruntime
import librosa
import numpy as np
from transformers import Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)
with open("./models/w2v2-base-250h/processor/vocab.json", 'r') as f:
    json_ = f.read()
    vocab_dict = json.loads(json_)
    print(vocab_dict.items())
    vocab_dict = sorted(vocab_dict.items(), key=lambda x: int(x[1]))
    print(vocab_dict)
vocab_list = [i[0] for i in vocab_dict]
# vocab_list = sorted(vocab_dict)
print(vocab_list)
decoder = build_ctcdecoder(
    vocab_list,
    kenlm_model_path='./models/w2v2-base-250h/processor/vi_lm_4grams.bin',  # either .arpa or .bin file
    # unigrams=hotwords,
    alpha=0.5,  # tuned on a val set
    beta=1.5,  # tuned on a val set
)

print(onnxruntime.get_device())

processor = Wav2Vec2Processor.from_pretrained("./models/w2v2-base-250h/processor")
data, sr = librosa.load("./sample/haimy_voice.wav", sr=16000)

sess = onnxruntime.InferenceSession('./models/w2v2-base-250h/model/model.onnx',providers=['CUDAExecutionProvider'] )
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
outputs = [x.name for x in sess.get_outputs()]
print(input_name, output_name)
y = sess.run(outputs, {sess.get_inputs()[0].name: np.expand_dims(data, axis=0)})[0]

print(y.shape)

# logit = np.argmax(y, axis=-1)
# print(logit.shape)
# output = processor.batch_decode(logit)
# print(output[0])

output = decoder.decode(np.squeeze(y))
print(output)