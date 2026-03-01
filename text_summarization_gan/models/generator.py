import torch.nn as nn
from transformers import T5ForConditionalGeneration

class TextGenerator(nn.Module):
    """
    Generator Network for the GAN structure.
    Uses an Encoder-Decoder architecture (T5) to generate abstractive summaries.
    """
    def __init__(self, model_name='t5-small'):
        super(TextGenerator, self).__init__()
        # Pre-trained Transformer Encoder-Decoder Generator
        self.generator = T5ForConditionalGeneration.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None):
        """Standard Forward Pass for Training the MLE baseline"""
        return self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        )
        
    def generate(self, input_ids, **kwargs):
        """Helper to call autoregressive generation"""
        return self.generator.generate(input_ids, **kwargs)
