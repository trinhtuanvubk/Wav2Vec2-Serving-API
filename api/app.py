import os
import io
import base64
import sys
import torch
import numpy as np
import traceback
import uuid
import onnxruntime
import json
import gc
import tritonhttpclient
import librosa
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import APIRouter, HTTPException, status, File, Body, UploadFile, Request

from pydantic import BaseModel, Field, model_validator
from pyctcdecode import build_ctcdecoder

input_name = 'audio_input'
output_name = 'transcript_output'
VERBOSE = False
model_name = 'ensemble'
url = '0.0.0.0:8050'
model_version = '1'

triton_client = tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)   

MEDIA_ROOT = "./temp_media"
os.makedirs(MEDIA_ROOT, exist_ok=True)

class AudioInput(BaseModel):
    # data: np.array = Field(default_factory=lambda: np.zeros(10))
    data: list
    # sample_rate: int
    # class Config:
    #     arbitrary_types_allowed = True

class Base64Input(BaseModel):
    data: str
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

app = FastAPI(title="ASR-API")

origins = [
    "*"
]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

@app.get('/')
async def health_check():
    data = {"Status":200}
    return data

@app.post('/onnx/trans-by-array')
async def onnx_trans_arr(audio: AudioInput):
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

@app.post('/onnx/trans-by-base64')
async def onnx_trans_base64(audio: Base64Input):
    try:
        decode_string = base64.b64decode(audio.data)
        # decode_string = audio.data.decoce("base64")
        arr = librosa.load(io.BytesIO(decode_string), sr=16000)
        
        input_data = np.array(arr[0], dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        logits = sess.run(output_names, {sess.get_inputs()[0].name: input_data})[0]
        # prediction = np.argmax(logits, axis=-1)
        transcript = decoder.decode(logits.squeeze())
        gc.collect()
        return {'transcript': transcript}

    except Exception as e:
            print(traceback.format_exception(None, e, e.__traceback__), file=sys.stderr, flush=True)
            raise HTTPException(status_code=400, detail="Audio bytes expected")
    
@app.post('/onnx/trans-by-file')
async def onnx_trans_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".wav", ".mp3")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File should be end with .jpg and .png"
        )
    # try:
    job_id = str(uuid.uuid4())
    # output_dir = os.path.join(MEDIA_ROOT, str(job_id))
    # if not os.path.exists(output_dir):
        # os.makedirs(output_dir)
    # audio_save_path = os.path.join(output_dir, file.filename)
    audio_save_path = os.path.join(MEDIA_ROOT, file.filename)

    with open(audio_save_path, "wb+") as file_object:
        file_object.write(file.file.read())

    data, rate = librosa.load(audio_save_path,sr=16000)
    # audio_duration = librosa.get_duration(data, sr=16000)
    # data = noise_reduce(data,rate)
    # temp_data = data.tolist()
    audio_data = np.expand_dims(data, axis=0)
    input_data = audio_data.astype('float32')

    # decode_string = base64.b64decode(audio.data)
    # # decode_string = audio.data.decoce("base64")
    # arr = librosa.load(io.BytesIO(decode_string), sr=16000)
    
    # input_data = np.array(arr[0], dtype=np.float32)
    # input_data = np.expand_dims(input_data, axis=0)
    logits = sess.run(output_names, {sess.get_inputs()[0].name: input_data})[0]
    # prediction = np.argmax(logits, axis=-1)
    transcript = decoder.decode(logits.squeeze())
    gc.collect()
    return {'transcript': transcript}

    # except Exception as e:
    #         print(traceback.format_exception(None, e, e.__traceback__), file=sys.stderr, flush=True)
    #         raise HTTPException(status_code=400, detail="Audio bytes expected")

@app.post('/onnx/trans-by-file-params')
async def onnx_trans_filexx(request: Request):
    content_type = request.headers.get("content-type")
    if content_type == "application/x-www-form-urlencoded":
        form_data = await request.form()
        if "file" not in form_data:
            return {"message": "Invalid input(request body).", "code": 34}
        
        audio_save_path = form_data["file"]
   
        # audio_save_path = os.path.join(MEDIA_ROOT, audio_save_path)

        # with open(audio_save_path, "wb+") as file_object:
        #     file_object.write(file.file.read())

        data, rate = librosa.load(audio_save_path,sr=16000)

        audio_data = np.expand_dims(data, axis=0)
        input_data = audio_data.astype('float32')

        logits = sess.run(output_names, {sess.get_inputs()[0].name: input_data})[0]
        # prediction = np.argmax(logits, axis=-1)
        transcript = decoder.decode(logits.squeeze())
        logger.debug("trans:{}".format(transcript))
        gc.collect()
        return {'transcript': transcript}

@app.post('/triton/trans-by-array')
async def triton_func_arr(audio: AudioInput):
    try:
        input_data = np.array(audio.data, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        input = tritonhttpclient.InferInput(input_name, input_data.shape, 'FP32')
        input.set_data_from_numpy(input_data)
        output = tritonhttpclient.InferRequestedOutput(output_name)    

        response_triton = triton_client.infer(model_name, model_version=model_version, inputs=[input], outputs=[output])
        raw_transcript = response_triton.as_numpy('transcript_output')
        final_transcript = raw_transcript[0].decode("utf-8")
        gc.collect()
        return {'transcript': final_transcript}

    except Exception as e:
            print(traceback.format_exception(None, e, e.__traceback__), file=sys.stderr, flush=True)
            raise HTTPException(status_code=400, detail="Audio bytes expected")
    
@app.post('/triton/trans-by-base64')
async def triton_func_base64(audio: Base64Input):
    try:
        decode_string = base64.b64decode(audio.data)
        # decode_string = audio.data.decoce("base64")
        arr = librosa.load(io.BytesIO(decode_string), sr=16000)
        
        input_data = np.array(arr[0], dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        
        input = tritonhttpclient.InferInput(input_name, input_data.shape, 'FP32')
        input.set_data_from_numpy(input_data)
        output = tritonhttpclient.InferRequestedOutput(output_name)    

        response_triton = triton_client.infer(model_name, model_version=model_version, inputs=[input], outputs=[output])
        raw_transcript = response_triton.as_numpy('transcript_output')
        final_transcript = raw_transcript[0].decode("utf-8")
        gc.collect()
        return {'transcript': final_transcript}

    except Exception as e:
            print(traceback.format_exception(None, e, e.__traceback__), file=sys.stderr, flush=True)
            raise HTTPException(status_code=400, detail="Audio bytes expected")
    


@app.post('/triton/trans-by-file-params')
async def triton_trans_filexx(request: Request):
    content_type = request.headers.get("content-type")
    if content_type == "application/x-www-form-urlencoded":
        form_data = await request.form()
        if "file" not in form_data:
            return {"message": "Invalid input(request body).", "code": 34}
        
        audio_save_path = form_data["file"]
   
        # audio_save_path = os.path.join(MEDIA_ROOT, audio_save_path)

        # with open(audio_save_path, "wb+") as file_object:
        #     file_object.write(file.file.read())
        data, rate = librosa.load(audio_save_path,sr=16000)

        audio_data = np.expand_dims(data, axis=0)
        input_data = audio_data.astype('float32')

        input = tritonhttpclient.InferInput(input_name, input_data.shape, 'FP32')
        input.set_data_from_numpy(input_data)
        output = tritonhttpclient.InferRequestedOutput(output_name)    

        response_triton = triton_client.infer(model_name, model_version=model_version, inputs=[input], outputs=[output])
        raw_transcript = response_triton.as_numpy('transcript_output')
        final_transcript = raw_transcript[0].decode("utf-8")
        logger.debug("trans:{}".format(final_transcript))
        gc.collect()
        return {'transcript': final_transcript}