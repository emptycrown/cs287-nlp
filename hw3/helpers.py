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

# Class that NMT{Trainer/Evaluator} extends

class NMTModelUser(object):
    # Models is a list [Encoder, Decoder]
    def __init__(self, models, TEXT_SRC, TEXT_TRG, **kwargs):
        self._TEXT_SRC = TEXT_SRC
        self._TEXT_TRG = TEXT_TRG
        self.models = models
        self.cuda = kwargs.get('cuda', True) and \
                    torch.cuda.is_available()
        if self.cuda:
            print('Using CUDA...')
        else:
            print('CUDA is unavailable...')

    def get_src_and_trg(self, batch):
        src = torch.t(batch.src.data).contiguous()
        trg = torch.t(batch.trg.data).contiguous()
        return (src, trg)

    def zeros_hidden(self, batch_sz, model_num):
        return torch.zeros(self.models[model_num].num_layers, batch_sz,
                           self.models[model_num].num_layers)

    # Ok to have self.prev_hidden apply to encoder then decoder since
    # encoder all ends before decoder starts
    def prepare_hidden(self, batch_sz, zero_out=True, model_num=0):
        if (not self.prev_hidden is None) and (not zero_out):
            pre_hidden = self.prev_hidden
        else:
            pre_hidden = (self.zeros_hidden(batch_sz, model_num) \
                          for i in range(2))
        if self.cuda:
            pre_hidden = tuple(t.cuda() for t in pre_hidden)
        return tuple(autograd.Variable(t) for t in pre_hidden)

    # kwargs can contain zero_out, model_num for prepare_hidden
    def prepare_model_inputs(self, batch, **kwargs):
        if self.cuda:
            src, trg = tuple(t.cuda() for t in self.get_src_and_trg(batch))
        else:
            src, trg = self.get_src_and_trg(batch)

        # TODO: can comment this out (assuming it passes)
        assert batch.src.size(1) == batch.trg.size(1)
        var_hidden = self.prepare_hidden(batch.src.size(1), **kwargs)

        var_src = autograd.Variable(src)
        var_trg = autograd.Variable(trg)

        return (var_src, var_trg, var_hidden)

    def init_epoch(self):
        self.prev_hidden = None

    # Assume log_probs is [batch_sz, sent_len, V], output is
    # [batch_sz, sent_len]
    @staticmethod
    def nll_loss(log_probs, output, **kwargs):
        log_probs_rshp = log_probs.view(-1, log_probs.size(2))
        output_rshp = output.view(-1, output.size(2))
        sent_len = batch.size(1)
        return F.nll_loss(log_probs_rshp, output_rshp, **kwargs) * \
            sent_len

class NMTEvaluator(NMTModelUser):
    def __init__(self, models, TEXT_SRC, TEXT_TRG, **kwargs):
        super(NMTEvaluator, self).__init__(models, TEXT_SRC, TEXT_TRG,
                                           **kwargs)

    def evaluate(self, test_iter, num_iter=None):
        start_time = time.time()
        for model in self.models:
            model.eval()

        for i,batch in enumerate(test_iter):
            # var_src, var_trg are [batch_sz, sent_len]
            var_src, var_trg, var_hidden = self.prepare_model_inputs(batch, zero_out=True,
                                                                     model_num=0)

            # TODO: implement beam search!

    
class NMTTrainer(NMTModelUser):
    def __init__(self, models, TEXT_SRC, TEXT_TRG, **kwargs):
        super(NMTTrainer, self).__init__(models, TEXT_SRC, TEXT_TRG, **kwargs)

        self.use_attention = kwargs.get('attention', False)
        self.base_lrn_rate = kwargs.get('lrn_rate', 0.1)
        self.optimizer_type = kwargs.get('optimizer', optim.SGD)
        self.optimizers = [self.optimizer_type(filter(lambda p : p.requires_grad,
                                                      model.parameters()),
                                               lr = self.base_lrn_rate) for \
                           model in self.models]

        self.lr_decay_opt = kwargs.get('lrn_decay', 'none')
        # TODO: setup for lr decay

        self.clip_norm = kwargs.get('clip_norm', 5)
        self.init_lists()
        if self.cuda:
            for model in self.models:
                model.cuda()

    def init_lists(self):
        self.training_losses = list()
        self.training_norms = list()
        self.val_prefs = list()

    def get_loss_data(self, loss):
        if self.cuda:
            return loss.data.cpu().numpy()[0]
        else:
            return loss.data.numpy()[0]

    def make_recordings(self, loss, norm):
        self.training_norms.append(norm)
        self.training_losses.append(loss)

    def clip_norms(self):
        # Clip grad norm after backward but before step
        if self.clip_norm > 0:
            parameters = tuple()
            for model in self.models:
                parameters += model.parameters()
                
            # Norm clipping: returns a float
            norm = nn.utils.clip_grad_norm(
                parameters, self.clip_norm)
        else:
            norm = -1
        return norm

    def train_batch(self, batch):
        for model in self.models:
            model.zero_grad()

        # var_src, var_trg are [batch_sz, sent_len]
        var_src, var_trg, var_hidden = self.prepare_model_inputs(batch, zero_out=True,
                                                                 model_num=0)

        # For attention, will use enc_output (not otherwise)
        enc_output, enc_hidden = self.models[0](var_src, var_hidden)
        self.prev_hidden = enc_hidden
        if self.use_attention:
            raise NotImplementedError('Attention not yet implemented!')
        else:
            # Using real words as input. Use prev_hidden both to
            # initialize hidden state (the first time) and as context
            # vector
            dec_output, dec_hidden = self.models[0](var_trg, self.prev_hidden,
                                                    enc_hidden)
            self.prev_hidden = dec_hidden

            loss = self.nll_loss(dec_output, var_trg)

        loss.backward()

        # norms must be clipped after backward but before step
        norm = self.clip_norms()

        loss_data = self.get_loss_data(loss)
        if kwargs.get('verbose', False):
            self.make_recordings(loss_data, norm)

        for optimizer in self.optimizers:
            optimizer.step()

        # Return loss and norm (before gradient step)
        return loss_data, norm

    def init_parameters(self):
        for model in self.models:
            for p in model.parameters():
                p.data.uniform_(-0.05, 0.05)

    def train(torch_train_iter, le=None, val_iter=None, **kwargs):
        self.init_lists()
        start_time = time.time()
        self.init_parameters()
        torch_train_iter.init_epoch()
        for epoch in range(kwargs.get('num_iter', 100)):
            self.init_epoch()
            for model in self.models:
                model.train()

            # TODO: LR decay3
            train_iter = iter(torch_train_iter)

            for batch in train_iter:
                res_loss, res_norm = self.train_batch(batch)

            if epoch % kwargs.get('skip_iter', 1) == 0:
                if not kwargs.get('verbose', False):
                    self.make_recordings(res_loss, res_norm)

            print('Epoch %d, loss: %f, norm: %f, elapsed: %f, lrn_rate: %f' \
                  % (epoch, np.mean(self.training_losses[-10:]),
                     np.mean(self.training_norms[-10:]),
                     time.time() - start_time,
                     self.base_lrn_rate)) #  * self.lambda_lr(epoch)))
                    
            
            if (not le is None) and (not val_iter is None):
                self.val_perfs.append(le.evaluate(val_iter))
                print('Validation set metric: %f' % \
                      self.val_perfs[-1])

        if len(self.val_perfs) >= 1:
            print('FINAL VAL PERF', self.val_perfs[-1])
            return self.val_perfs[-1]
        return -1
