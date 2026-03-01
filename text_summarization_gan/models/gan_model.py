# This file acts as an orchestrator or aggregator, keeping the references for the GAN models logic.
from .generator import TextGenerator
from .discriminator import TextDiscriminator

def get_gan_components(gen_model_name='t5-small', disc_model_name='distilbert-base-uncased', device='cpu'):
    """
    Combines the Discriminator and Generator architecture required for the model.
    """
    generator = TextGenerator(gen_model_name).to(device)
    discriminator = TextDiscriminator(disc_model_name).to(device)
    
    return generator, discriminator
