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
class LinearIAFMLPEncoder(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=10, img_width=28,
                 img_height=28):
        super(LinearIAFMLPEncoder, self).__init__()
        self.img_size = img_width * img_height
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(self.img_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)
        # This is a little wasteful (factor of 2), but I can't figure
        # out a better way to do it
        self.linear4 = nn.Linear(hidden_dim, latent_dim**2)

        self.id_helper = V(torch.eye(latent_dim),
                           requires_grad=False).view(1, latent_dim, latent_dim)
        tril = torch.ones(latent_dim, latent_dim).tril()
        self.tril_helper = V(tril, requires_grad=False).view(1, latent_dim, latent_dim)

    def forward(self, x, view_as_img=True):
        if view_as_img:
            x = x.view(-1, self.img_size)
        h = F.relu(self.linear1(x))
        # Inverse Cholesky matrix:
        L = self.linear4(h).view(-1, self.latent_dim, self.latent_dim)
        L = torch.mul(L, self.tril_helper) # lower triangular part (broadcasting)

        # Ensure diagonal is all 1s; broadcasting again (in 2 ways)
        L = self.id_helper + torch.mul(1 - self.id_helper, L)
        return self.linear2(h), self.linear3(h), L

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

class IAFNormalVAE(NormalVAE):
    def __init__(self, *args):
        super(IAFNormalVAE, self).__init__(*args)
        
    def forward(self, x_src, enc_view_img=True, dec_view_img=True):
        mu, logvar, L = self.encoder(x_src, view_as_img=enc_view_img)
        q_normal = Normal(loc=mu, scale=logvar.mul(0.5).exp())
        y_sample = q_normal.rsample()
        
        # L is [batch_sz, latent_dim, latent_dim],
        # y_sample is [batch_sz, latent_dim]
        y_sample_rshp = y_sample.view(-1, self.encoder.latent_dim, 1)
        # sample is [batch_sz, latent_dim]
        z_sample = torch.bmm(L, y_sample_rshp).squeeze()
        x_reconstruct = self.decoder(z_sample, view_as_img=dec_view_img)
        
        q_probs = q_normal.log_prob(y_sample)
        # TODO: verify: since diagonal is 1, this is a
        # volume-preserving transofmration, so just compute
        # q_{old}(y|x) (= q_{new}(z|x))
        return (x_reconstruct, q_probs, z_sample)
    
class MLPGenerator(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=10, img_width=28,
                 img_height=28):
        super(MLPGenerator, self).__init__()
        self.img_size = img_width * img_height
        self.img_height = img_height
        self.img_width = img_width
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Linear(int(hidden_dim/2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.img_size))

    # Pixels are all in [0,1], so may as well apply sigmoid
    def forward(self, z, view_as_img=True):
        x = self.mlp(z)
        # x = self.linear3(F.relu(self.linear2(F.relu(self.linear1(z)))))
        if view_as_img:
            x = x.view(-1, self.img_height, self.img_width)
        return F.sigmoid(x)

class DeconvGenerator(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=10, img_width=28,
                 img_height=28, num_chans_start=5, dim_start=7, num_chans_mid=64,
                 dropout=0.3):
        super(DeconvGenerator, self).__init__()
        self.img_size = img_width * img_height
        self.img_height = img_height
        self.img_width = img_width
        self.latent_dim = latent_dim
        self.dim_start = dim_start
        self.dropout = dropout
        self.num_chans_start = num_chans_start
        self.num_chans_mid = num_chans_mid
        self.linear1 = nn.Linear(latent_dim, num_chans_start * (dim_start**2))
        self.conv_layers = {2 : [-1, num_chans_mid, 14, 14],
                            5 : [-1, num_chans_mid, 28, 28],
                            11 : [-1, 1, 28, 28]}
        self.deconv_net = nn.Sequential(
            # nn.Dropout(self.dropout),
            nn.BatchNorm2d(num_chans_start),
            nn.ReLU(),
            nn.ConvTranspose2d(num_chans_start, num_chans_mid, 5, stride=2,
                               padding=2, bias=False), # 6 * 2 - 2 * 2 + 5
            # nn.Dropout(self.dropout),
            nn.BatchNorm2d(num_chans_mid),
            nn.ReLU(),
            nn.ConvTranspose2d(num_chans_mid, 1, 5, stride=2,
                               padding=2, bias=False))  # 13 * 2 - 2 * 2 + 5
            # nn.Dropout(self.dropout),
            # nn.BatchNorm2d(num_chans_mid),
            # nn.ReLU(),
            # nn.ConvTranspose2d(num_chans_mid, 1, 5, stride=1,
            #                    padding=2)) # 27 * 1 - 2 * 2 + 5
            

    # Pixels are all in [0,1], so may as well apply sigmoid
    # TODO: this is new, untested!
    def forward(self, z, view_as_img=True):
        x = self.linear1(z).view(-1, self.num_chans_start, self.dim_start,
                                     self.dim_start)
        for i in range(len(self.deconv_net)):
            if i in self.conv_layers:
                x = self.deconv_net[i](x, output_size=self.conv_layers[i])
            else:
                x = self.deconv_net[i](x)                
                
        if not view_as_img:
            x = x.view(-1, self.img_size)

        # squeeze the output channel
        return F.sigmoid(x.squeeze())

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

class ConvDiscriminator(nn.Module):
    def __init__(self, latent_dim=10, img_width=28, img_height=28,
                 num_chans_mid=64, dim_end=7, num_chans_end=10,
                 hidden_dim=0, kernel_size=4, dropout=0.3):
        super(ConvDiscriminator, self).__init__()
        self.img_size = img_width * img_height
        self.img_width = img_width
        self.img_height = img_height
        self.dropout = dropout
        self.num_chans_mid = num_chans_mid
        self.num_chans_end = num_chans_end
        self.dim_end = dim_end
        self.latent_dim = latent_dim
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, num_chans_mid, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            # nn.Dropout(self.dropout),
            nn.Conv2d(num_chans_mid, num_chans_end, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_chans_end),
            nn.LeakyReLU(0.2))
            # nn.Dropout(self.dropout),
            # nn.Conv2d(num_chans_mid, num_chans_end, 4, stride=2, padding=1, bias=False),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(self.dropout),
            # nn.BatchNorm2d(num_chans_end))
        self.linear = nn.Linear(dim_end**2 * num_chans_end, 1)

    def forward(self, x, view_as_img=True):
        # no matter the value of view_as_img, need to reshape to add
        # channel dim
        x = x.view(-1, 1, self.img_height, self.img_width)
        x = self.conv_net(x).view(-1, self.num_chans_end * (self.dim_end**2))
        return F.sigmoid(self.linear(x))
