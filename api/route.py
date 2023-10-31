import os
import sys
import torch
import numpy as np
import traceback
import onnxruntime
import json
import gc
from fastapi import APIRouter, HTTPException, status, File, Body

from pydantic import BaseModel, Field, model_validator
from pyctcdecode import build_ctcdecoder

class AudioInput(BaseModel):
    data: np.array = Field(default_factory=lambda: np.zeros(10))
    sample_rate: int
    class Config:
        arbitrary_types_allowed = True
    
# load vocab
with open("./models/w2v2-base-250h/processor/vocab.json", 'r') as f:
    json_ = f.read()
    vocab_dict = json.loads(json_)
    vocab_dict = sorted(vocab_dict.items(), key=lambda x: int(x[1]))
vocab_list = [i[0] for i in vocab_dict]

# pyctc decoder
decoder = build_ctcdecoder(
    vocab_list,
    kenlm_model_path='./models/w2v2-base-250h/processor/vi_lm_4grams.bin',  # either .arpa or .bin file
    # unigrams=hotwords,
    alpha=1.0,  # tuned on a val set
    beta=1.5,  # tuned on a val set
)

# load session
sess = onnxruntime.InferenceSession('./models/w2v2-base-250h/model/model.onnx',providers=['CUDAExecutionProvider'] )
output_names = [x.name for x in sess.get_outputs()]
# router definition
router = APIRouter(prefix="/ocr", tags=["ASR"])

router.post('/onnx/transcribe')
async def onnx_transcribe(audio: AudioInput):
    try:
        input_data = np.array(audio.data, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        logits = sess.run(output_names, {sess.get_inputs()[0].name: input_data})[0]
        # prediction = np.argmax(logits, axis=-1)
        transcript = decoder.decode(logits.squeeze())
        gc.collect()
        return {'transcript': transcript}

    except Exception as e:
            print(traceback.format_exception(None, e, e.__traceback__), file=sys.stderr, flush=True)
            raise HTTPException(status_code=400, detail="Audio bytes expected")


router.post('/triton/transcribe')
async def triton_transcribe(audio: AudioInput):
    pass
