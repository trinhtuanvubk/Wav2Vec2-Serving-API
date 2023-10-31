import numpy as np
import sys
import os
import json
from pathlib import Path

import triton_python_backend_utils as pb_utils

from pyctcdecode import build_ctcdecoder

NGRAM = "ngram/vi_lm_4grams.bin"
VOCAB = "ngram/vocab.json"
cur_folder = Path(__file__).parent

ngram_path = str(cur_folder/NGRAM)
vocab_path = str(cur_folder/VOCAB)


class TritonPythonModel:
    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args['model_config']), "transcript"
            )['data_type']
        )
        with open(vocab_path, 'r', encoding='utf-8') as f:
            json_ = f.read()
            vocab_dict = json.loads(json_)
            # sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
            # vocab_dict = sorted(vocab_dict.items(), key=lambda x: int(x[1]))
        # self.vocab_list = [i[0] for i in sort_vocab]
        self.vocab_list = sorted(vocab_dict)
        self.decoder = build_ctcdecoder(
                                self.vocab_list,
                                kenlm_model_path=ngram_path,  # either .arpa or .bin file
                                alpha=0.5,  # tuned on a val set
                                beta=1.5,  # tuned on a val set
                            )
        
        print("TritonPythonModel initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "logits")
            logits = logits.as_numpy()

            transcript = self.decoder.decode(logits.squeeze())

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "transcript",
                        transcript.astype(self.output_dtype)
                    )
                ]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')