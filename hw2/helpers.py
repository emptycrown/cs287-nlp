import torchtext
from torchtext.vocab import Vectors
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools as it


class LangEvaluator(object):
    def __init__(self, model, TEXT, **kwargs):
        self._TEXT = TEXT
        self.model = model
        self.eval_metric = kwargs.get('evalmetric', 'perplexity')
        
    def evaluate(self, test_iter, num_iter=None):
        sum_nll = 0
        cnt_nll = 0
        for i,batch in enumerate(test_iter):
            if i % 100 == 0:
                print('Iteration %d' % i)
            # Model output: [batch_size, sent_len, size_vocab]; these
            # aren't actually probabilities if the model is a Trigram,
            # but this doesn't matter.
            batch_transpose = torch.t(batch.text).contiguous() # [batch_size, sent_len]
            log_probs = self.model(batch_transpose)
            cnt_nll += batch_transpose.size()[0] * batch_transpose.size()[1]
            sum_nll += LangTrainer.loss_nll(batch_transpose.data,
                                            log_probs, mode='sum')
            if not num_iter is None and i > num_iter:
                break
            
        return np.exp(sum_nll / cnt_nll)

class LangTrainer(object):
    def __init__(self, TEXT, model, **kwargs):
        # Settings:
        optimizer = kwargs.get('optimizer', optim.SGD)
        self._optimizer = optimizer(filter(lambda p : p.requires_grad,
                                           model.parameters()),
                                    lr=kwargs.get('lr', 0.1))        
        self.cuda = kwargs.get('cuda', True) and \
            torch.cuda.is_available()
        if self.cuda:
            print('Using CUDA...')
        self.clip_norm = kwargs.get('clip_norm', 5)
            
        self._TEXT = TEXT
        self.model = model
        # TODO: implement validation thing for early stopping
        self.training_losses = list()
        self.training_norms = list()
        if self.cuda:
            self.model.cuda()
    
    # Here batch is output from a RNN/NNLM/Trigram model:
    # [..., size_vocab], and output are the real words: [...]
    @staticmethod
    def loss_nll(batch, output, mode='mean'):
        # [batch_size * sent_len, size_vocab]
        vocab_len = output.size()[-1]
        output = output.view(-1, vocab_len)
        # [batch_size * sent_len]
        batch = batch.view(-1, 1)
        batch_probs = -1 * torch.gather(output, 1, 
                                        batch) #.type(torch.LongTensor))
        if mode == 'mean':
            return torch.mean(batch_probs)
        else:
            return torch.sum(batch_probs)
        return
    
    @staticmethod
    def loss_perplexity(*args):
        return torch.exp(self.loss_nll(*args))
    
    def get_feature(self, batch):
        return torch.t(batch.text.data).contiguous()

    # The labels we use as the true words: same as features
    def get_label(self, batch):
        return self.get_feature(batch)
    
    # We are doing a slightly funky thing of taking a 
    # variable's data and then making a new 
    # variable...this seems cleaner though
    def make_loss(self, batch):
        if self.cuda:
            feature, label = self.get_feature(batch).cuda(), \
                            self.get_label(batch).cuda()
        else:
            feature, label = self.get_feature(batch), \
                            self.get_label(batch)
        var_feature = autograd.Variable(feature)
        var_label = autograd.Variable(label)
        loss = self.loss_nll(var_label, self.model(var_feature))
        return loss
    
    def train(self, train_iter, **kwargs):
        train_iter = iter(train_iter)
        for i in range(kwargs.get('num_iter', 100)):
            batch = next(train_iter)
            self.model.zero_grad()
            loss = self.make_loss(batch)
            self.training_losses.append(loss.data.numpy()[0])

                
                
            # Norm clipping: returns a float
            norm = nn.utils.clip_grad_norm(filter(lambda p : p.requires_grad,
                                                  self.model.parameters()), self.clip_norm)
            self.training_norms.append(norm)    
            if i % kwargs.get('skip_iter', 10) == 0:
                print('Iteration %d, loss: %f, norm: %f' % (i, self.training_losses[-1],
                                                            self.training_norms[-1]))
            # Do gradient updates
            loss.backward()
            self._optimizer.step()
            
