import torch
import soundfile
import torchaudio
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
from utils import wav2vec2_layer

class W2V2:
    def __init__(self, model_path, processor_path, withLM=False):
        self.model_path = model_path
        self.processor_path = processor_path
        self.withLM = withLM
        self.model, self.processor = self.load_model()

    def load_model(self):
        if self.withLM:
            # model_name = "nguyenvulebinh/wav2vec2-base-vi-vlsp2020"
            model = wav2vec2_layer.Wav2Vec2ForCTC.from_pretrained(self.model_path, local_files_only=True)
            processor = Wav2Vec2ProcessorWithLM.from_pretrained(self.processor_path, local_files_only=True)
        else:
            model = Wav2Vec2ForCTC.from_pretrained(self.model_path,  local_files_only=True)
            processor = Wav2Vec2Processor.from_pretrained(self.processor_path,  local_files_only=True)
        return model, processor
    
    def transcribe(self, audio, sampling_rate=16000, useLM=False):
        input_data = self.processor.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors='pt')
        output = self.model(input_data.input_values)

        if useLM:
            transcripts = self.processor.decode(output.logits.cpu().detach().numpy()[0], beam_width=100).text
        else:
            transcripts = self.processor.tokenizer.decode(output.logits.argmax(dim=-1)[0].detach().cpu().numpy())

        return transcripts

