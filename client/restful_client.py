import requests
import librosa
import numpy as np
import json


def restful_client(path, url):
    # speech_array, sr = sf.read(path)
    speech_array, sr = librosa.load(path, sr=16000)
    # speech_array = np.expand_dims(speech_array, axis=0)
    print(speech_array.shape)
    speech_array = speech_array.tolist()
    payload = {"data": speech_array,
               }
    res = requests.post(url, json = payload)
    trans = json.loads(res.text)
    print(trans)





if __name__=="__main__":
    path = "./sample/haimy_voice.wav"
    url = "http://0.0.0.0:1445/onnx/transcribe/"
    restful_client(path, url)

