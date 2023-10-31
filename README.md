# Wav2Vec2 API 

### Download base model
- Download model via this link: https://huggingface.co/nguyenvulebinh/wav2vec2-large-vi-vlsp2020

### Export model to ONNX

- To export:
```
python3 export_onnx.py
```
### Triton
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

### Restful API

- To run restful API:
```
cd api
uvicorn api.app:app --host 0.0.0.0 --port 8070
```