import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# New stuff
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

# Compute the variational parameters for q
class MLPEncoder(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=10, img_width=28,
                 img_height=28):
        super(Encoder, self).__init__()
        self.img_size = img_width * img_height
        self.linear1 = nn.Linear(self.img_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, view_as_img=True):
        if view_as_img:
            x = x.view(-1, img_size)
        h = F.relu(self.linear1(x))
        return self.linear2(h), self.linear3(h)

# Implement the generative model p(x | z)
class MLPDecoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__(hidden_dim=200, latent_dim=10, img_width=28,
                                      img_height=28)
        self.img_size = img_width * img_height
        self.img_height = img_height
        self.img_width = img_width
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.img_size)

    def forward(self, z, view_as_img=True):
        x = self.linear2(F.relu(self.linear1(z)))
        if view_as_img:
            x = x.view(-1, self.img_height, self.img_width)
        return x

# VAE using reparameterization "rsample"
class NormalVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(NormalVAE, self).__init__()

        # Parameters phi and computes variational parameters lambda
        self.encoder = encoder

        # Parameters theta, p(x | z)
        self.decoder = decoder
    
    def forward(self, x_src, enc_view_img=True, dec_view_img=True):
        # Example variational parameters lambda
        mu, logvar = self.encoder(x_src, view_as_img=enc_view_img)

        # [batch_sz, img_ht, img_wdth]; TODO: why * 0.5?
        q_normal = Normal(loc=mu, scale=logvar.mul(0.5).exp())

        # Reparameterized sample.
        z_sample = q_normal.rsample()
        #z_sample = mu
        return self.decoder(z_sample, view_as_img=dec_view_img), q_normal