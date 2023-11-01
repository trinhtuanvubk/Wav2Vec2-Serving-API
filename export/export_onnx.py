import torch
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def convert_torch_to_onnx_batch(model_path, output_path, dummy_input, device=None):
    
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    
    input_names = ["audio"]
    output_names = ["logits"]
    
    if device!=None:
        model = model.to(device)
        dummy_input = dummy_input.to(device)
    
    torch.onnx.export(model, 
                 dummy_input, 
                 output_path, 
                 verbose=True, 
                 input_names=input_names, 
                 output_names=output_names,
                 dynamic_axes={'audio' : {1 : 'audio_len'},    # variable length axes
                               'logits' : {1 : 'audio_len'}})
    

if __name__=="__main__":

    device = torch.device('cuda')
    print(device)
    output_path = "../models/w2v2-base-250h/model/model.onnx"
    model_path = "../models/w2v2-base-250h/model/"

    dummy_input = torch.rand([1,2500])
    
    convert_torch_to_onnx_batch(model_path,output_path,dummy_input,device=device)

