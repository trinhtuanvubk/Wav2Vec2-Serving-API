# Wav2Vec2 API 
- This repo builds Wav2Vec2 API service with both restful API and Triton API.

### Download base model
- Download model via this link: https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h

### Export model to ONNX

- To export:
```
cd export
python3 export_onnx.py
```

- To test onnx model:
```
python3 onnx_test.py
```

### Run API
###### Triton API
- To build a wrapper image of triton:
```
cd triton_deploy
docker build -t triton_with_pyctcdecode .
```

- To run triton server:
```
docker run --gpus=1 -itd \
--add-host=host.docker.internal:host-gateway \
-p 8050-8052:8000-8002 \
-v ${PWD}/model_repository:/models \
--name triton_deploy triton_with_pyctcdecode:latest \
tritonserver --model-repository=/models
``` 

- Note:
    - copy `model.onnx` into `model_repository/wav2vec2/1/model.onnx`
    - copy `vi_lm_4grams.bin` into `model_repository/decoder/1/ngram/vi_lm_4grams.bin` 
##### Restful API

- To run restful API:
```
uvicorn api.app:app --host 0.0.0.0 --port 1445
```

### Client
```
cd client
```
###### Triton client
```
python3 triton_client.py
```
###### Restful client
```
python3 restful_client.py
```