import torchtext
from torchtext.vocab import Vectors
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools as it
import time

class LangEvaluator(object):
    def __init__(self, model, TEXT, **kwargs):
        self._TEXT = TEXT
        self.model = model
        self.eval_metric = kwargs.get('evalmetric', 'perplexity')
        self.cuda = kwargs.get('cuda', True) and \
            torch.cuda.is_available()
        if self.cuda:
            print('Using CUDA for evaluation...')
        
    def evaluate(self, test_iter, num_iter=None, use_variable=True):
        start_time = time.time()
        self.model.eval() # In case we have dropout
        sum_nll = 0
        cnt_nll = 0
        for i,batch in enumerate(test_iter):
            # Model output: [batch_size, sent_len, size_vocab]; these
            # aren't actually probabilities if the model is a Trigram,
            # but this doesn't matter.
            
            # TODO: this might break trigram stuff (easy to fix)...
            # Also, can probably combine with LangTrainer stuff
            if self.cuda:
                # [batch_size, sent_len]                
                feature, label = LangTrainer.get_feature(batch).cuda(), \
                                 LangTrainer.get_label(batch).cuda()
            else:
                feature, label = LangTrainer.get_feature(batch), \
                                 LangTrainer.get_label(batch)
            var_feature = autograd.Variable(feature)
            var_label = autograd.Variable(label)
            log_probs = self.model(var_feature)
            cnt_nll += var_feature.data.size()[0] * \
                       var_feature.data.size()[1]
            sum_nll += LangTrainer.loss_nll(var_feature,
                                            log_probs, mode='sum').data[0]
            if not num_iter is None and i > num_iter:
                break

        self.model.train() # Wrap model.eval()
        print('Validation time: %f seconds' % (time.time() - start_time))
        return np.exp(sum_nll / cnt_nll)

class LangTrainer(object):
    def __init__(self, TEXT, model, **kwargs):
        # Settings:
        self.base_lrn_rate = kwargs.get('lrn_rate', 0.1)
        optimizer = kwargs.get('optimizer', optim.SGD)
        self._optimizer = optimizer(filter(lambda p : p.requires_grad,
                                           model.parameters()),
                                    lr=self.base_lrn_rate)

        # Do learning rate decay:
        lr_decay_opt = kwargs.get('lrn_decay', 'none')
        if lr_decay_opt == 'none':
            self.lambda_lr = lambda i : 1
        elif lr_decay_opt == 'invlin':
            decay_rate = kwargs.get('lrn_decay_rate', 0.001)
            self.lambda_lr = lambda i : 1 / (1 + i * decay_rate)
        else:
            raise ValueError('Invalid learning rate decay option: %s' \
                             % lr_decay_opt)
        self.scheduler = optim.lr_scheduler.LambdaLR(self._optimizer,
                                                     self.lambda_lr)
            
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
        self.val_perfs = list()
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

    @staticmethod
    def get_feature(batch):
        return torch.t(batch.text.data).contiguous()

    # The labels we use as the true words: same as features
    @staticmethod
    def get_label(batch):
        return LangTrainer.get_feature(batch)
    
    # We are doing a slightly funky thing of taking a 
    # variable's data and then making a new 
    # variable...this is kinda unnecessary and ugly
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

    # le is a LangEvaluator, and if supplied must have a val_iter
    # along with it
    def train(self, torch_train_iter, le=None, val_iter=None,
              **kwargs):
        start_time = time.time()
        for epoch in range(kwargs.get('num_iter', 100)):
            # Learning rate decay, if any
            self.scheduler.step()
            train_iter = iter(torch_train_iter)

            for batch in train_iter:
                self.model.zero_grad()
                loss = self.make_loss(batch)

                if self.clip_norm > 0:
                    # Norm clipping: returns a float
                    norm = nn.utils.clip_grad_norm(
                        filter(lambda p : p.requires_grad,
                               self.model.parameters()), self.clip_norm)
                    self.training_norms.append(norm)
                else:
                    self.training_norms.append(-1)
                    
                # Do gradient updates
                loss.backward()
                self._optimizer.step()

            # Logging, early stopping
            if epoch % kwargs.get('skip_iter', 1) == 0:
                # Update training_losses
                if self.cuda:
                    self.training_losses.append(loss.data.cpu().numpy()[0])
                else:
                    self.training_losses.append(loss.data.numpy()[0])

                # Logging
                print('Epoch %d, loss: %f, norm: %f, elapsed: %f, lrn_rate: %f' \
                      % (epoch, self.training_losses[-1],
                         self.training_norms[-1],
                         time.time() - start_time,
                         self.base_lrn_rate * self.lambda_lr(epoch)))
                if (not le is None) and (not val_iter is None):
                    self.val_perfs.append(le.evaluate(val_iter))
                    print('Validation set metric: %f' % \
                          self.val_perfs[-1])


