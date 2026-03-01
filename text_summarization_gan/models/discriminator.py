import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertConfig

class TextDiscriminator(nn.Module):
    """
    Discriminator Network for the GAN structure.
    Classifies sequences as Real (human-written summaries) or Fake (GAN-generated).
    """
    def __init__(self, model_name='distilbert-base-uncased'):
        super(TextDiscriminator, self).__init__()
        # 1 output class -> Logit representing 'Real' confidence
        config = DistilBertConfig.from_pretrained(model_name, num_labels=1)
        self.discriminator = DistilBertForSequenceClassification.from_pretrained(model_name, config=config)
        
    def forward(self, input_ids, attention_mask=None):
        """Outputs logits classifying sequence as Real (1) or Fake (0)"""
        outputs = self.discriminator(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
