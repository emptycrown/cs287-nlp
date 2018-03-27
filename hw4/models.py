import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import numpy as np

# New stuff
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

# Compute the variational parameters for q
class MLPEncoder(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=10, img_width=28,
                 img_height=28):
        super(MLPEncoder, self).__init__()
        self.img_size = img_width * img_height
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(self.img_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, view_as_img=True):
        if view_as_img:
            x = x.view(-1, self.img_size)
        h = F.relu(self.linear1(x))
        return self.linear2(h), self.linear3(h)

# Implement the generative model p(x | z)
class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=10, img_width=28,
                 img_height=28):
        super(MLPDecoder, self).__init__()
        self.img_size = img_width * img_height
        self.img_height = img_height
        self.img_width = img_width
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.img_size)

    def forward(self, z, view_as_img=True):
        x = self.linear2(F.relu(self.linear1(z)))
        if view_as_img:
            x = x.view(-1, self.img_height, self.img_width)
        return x

# Fancy posterior
class LinearIAFEncoder(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=10, img_width=28,
                 img_height=28):
        super(MLPEncoder, self).__init__()
        self.img_size = img_width * img_height
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(self.img_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)
        # This is a little wasteful (factor of 2), but I can't figure
        # out a better way to do it
        self.linear4 = nn.Linear(hidden_dim, latent_dim**2)

        self.id_helper = V(torch.eye(latent_dim),
                           requires_grad=False)

    def forward(self, x, view_as_img=True):
        if view_as_img:
            x = x.view(-1, self.img_size)
        h = F.relu(self.linear1(x))
        # Inverse Cholesky matrix:
        L = self.linear4(h).view(-1, self.latent_dim, self.latent_dim)
        L = L.tril() # lower triangular part

        # Ensure diagonal is all 1s
        L = self.id_helper + torch.mul(1 - self.id_helper, L)
        return self.linear2(h), self.linear3(h), L

class IAFNormalVAE(NormalVAE):
    def __init__(self, *args):
        super(IAFNormalVAE, self).__init__(*args)
        
    def forward(self, x_src, enc_view_img=True, dec_view_img=True,
                sample=True)
    vae = super(IAFNormalVAE)
    y_sample, q_normal = vae(x_src, enc_view_img, dec_view_img)
    if sample:
        return torch.matmul(L, y_sample) # TOOD: verify (3-tensor
                                         # times 2-tensor)
    else:
        return q_normal.log_prob(y_sample)
        # TODO: verify: since diagonal is 1, this is a
        # volume-preserving transofmration, so just compute
        # q_{old}(y|x) (= q_{new}(z|x))

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
    
class MLPGenerator(MLPDecoder):
    def __init__(self, **kwargs):
        super(MLPGenerator, self).__init__(**kwargs)

# TODO: set things right here        
class MLPDiscriminator(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=10, img_width=28,
                 img_height=28):
        super(MLPDiscriminator, self).__init__()
        self.img_size = img_width * img_height
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(self.img_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, view_as_img=True):
        if view_as_img:
            x = x.view(-1, self.img_size)
        h = F.relu(self.linear1(x))
        return F.sigmoid(self.linear2(h))
