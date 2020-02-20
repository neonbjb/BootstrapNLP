import transformers
import numpy as np

class DataProcessor:
    def __init__(self):
        # Configured to pre-process inputs to a sentiment predictor for GPT2.
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = "!"
        self.max_sequence_length = 128

    def process_input(self, inputs):
        return np.asarray(self.tokenizer.encode_plus(inputs["review"], add_special_tokens=True, max_length=self.max_sequence_length, pad_to_max_length=True)["input_ids"])

    def process_output(self, outputs):
        float_response = outputs['final_linear'].float_val[0]
        int_response = round(float_response)
        if int_response < 1:
            int_response = 1
        if int_response > 5:
            int_response = 5
        return {'star_rating': int_response}
