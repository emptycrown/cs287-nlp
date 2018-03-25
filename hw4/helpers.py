import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim

# New stuff.
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


# Functions to save/load models
def save_checkpoint(mod_enc, mod_dec, filename='checkpoint.pth.tar'):
    state_dict = {'model_encoder' : mod_enc.state_dict(),
                  'model_decoder' : mod_dec.state_dict()}
    torch.save(state_dict, filename)
def load_checkpoint(filename='checkpoint.pth.tar'):
    state_dict = torch.load(filename)
    return state_dict['model_encoder'], state_dict['model_decoder']
def set_parameters(model, sv_model, cuda=True):
    for i,p in enumerate(model.parameters()):
        p.data = sv_model[list(sv_model)[i]]
    model.cuda()

class LatentModelUser(object):
    # Model order: [encoder, decoder, [VAE]]
    def __init__(self, models, batch_sz, mode='vae', cuda=True):
        self.models = models
        self.cuda = cuda and torch.cuda.is_available()
        self.mode = mode
        self.bce_loss = nn.BCEWithLogitsLoss(size_average=False)        
        if self.cuda:
            print('Using CUDA...')
        else:
            print('CUDA is unavailable...')

        self.batch_sz = batch_sz
        if self.mode == 'vae':
            # Set prior
            img_size = self.models[0].img_size
            self.prior = Normal(V(torch.zeros(self.batch_sz, img_size)), 
                                V(torch.ones(self.batch_sz, img_size)))

    def run_model(self, batch, train=True, batch_avg=True):
        if train:
            for model in self.models:
                model.zero_grad()

        x = V(torch.FloatTensor(batch))
        out, q = self.models[2](x, enc_view_img=True, dec_view_img=False)

        # This is the NLL of the images according to the generative model out
        loss = self.bce_loss(out, x) # x plays role of target
        if batch_avg:
            loss = loss / self.batch_sz
        # Important fact: kl divergence is additive for product
        # distributions, so we can do the KL divergence batch by batch
        kl = kl_divergence(q, self.prior)
        if batch_avg:
            kl = kl / self.batch_sz
        return loss, kl

class LatentModelEvaluator(LatentModelUser):
    def __init__(self, models, cuda=True):
        super().__init__(models, cuda)

    def evaluate(self, val_loader, num_iter=None):
        start_time = time.time()
        for model in self.models:
            model.eval()

        loss_sum = 0
        kl_sum = 0
        data_cnt = 0

        # Since this is a conditionally conjugate model with no global
        # latent variables (i.e. only latent variables) and the KL
        # divergences are between product distributions (for both
        # prior and conditional; i.e. it is a random field), we can
        # compute KL divergence and NLL in batches
        for i,batch in val_loader:
            loss, kl = self.run_model(batch, train=False, batch_avg=False)
            loss_sum += loss.data[0]
            kl_sum += kl.data[0]
            data_cnt = batch.size(0)

            if i >= num_iter:
                break

        loss_avg = loss_sum / data_cnt
        kl_avg = kl_sum / data_cnt

        return loss_avg, kl_avg
        
            

class LatentModelTrainer(LatentModelUser):
    def __init__(self, models, cuda=True, lrn_rate=0.05, optimizer=optim.SGD):
        super().__init__(models, cuda)
        self.base_lrn_rate = lrn_rate
        self.optimizer_type = optimizer
        self.init_optimizers()
        if self.cuda:
            for model in self.models:
                model.cuda()

    def init_optimizers(self):
        if self.mode == 'vae':
            self.optimizer = self.optimizer(filter(lambda p : p.requires_grad,
                                                    models[2].parameters()),
                                            lr = self.base_lrn_rate)
        elif self.mode == 'gan':
            self.optimizers = [self.optimizer(filter(lambda p : p.requires_grad,
                                                     model.parameters()),
                                              lr = self.base_lrn_rate) for \
                               model in self.models]
        
    def init_parameters(self):
        for model in self.models:
            for p in model.parameters():
                p.data.uniform_(-0.05, 0.05)

        
    def init_lists(self):
        self.training_losses = list()
        self.training_kls = list()
        self.val_losses = list()
        self.val_kls = list()


    def train(self, train_loader, save_model_fn=None, le=None, val_loader=None,
              init_parameters=True, num_epochs=100, **kwargs):
        start_time = time.time()
        self.init_lists()
        if init_parameters:
            self.init_pararmeters()

        for epoch in range(num_epochs):
            for model in self.models:
                model.train()

            self.training_losses.append(0)
            self.training_kls.append(0)
            for i, batch in train_loader:
                loss, kl = self.run_model(batch, **kwargs))
                loss_comb = loss + kl
                loss_comb.backward()
                self.optimizer.step()

                print(loss.data)
                self.training_losses[-1] += loss.data[0]
                self.training_kls[-1] += kl.data[0]
            self.training_losses[-1] /= len(train_loader)
            self.training_kls[-1] /= len(train_loader)

        print('Epoch %d, loss: %f, KL: %f, lrn_rate: %f' \
              % (epoch, self.training_losses[-1], self.training_kls[-1],
                 self.base_lrn_rate))

        if (not le is None) and (not val_loader is None):
            val_loss, val_kl = le.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_kls.append(val_kl)

            print('Validation set: loss: %f, KL: %f' \
                  % (val_loss, val_kl))


        if not save_model_fn is None:
            pathname = 'saved_models/' + save_model_fn + \
                       '.epoch_%d.ckpt.tar' % epoch
            print('Saving model to %s' % pathname)
            save_checkpoint(self.models[0], self.models[1],
                       pathname)
        
                
                
        
