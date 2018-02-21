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

    # Assume log_probs is [batch_sz, sent_len, V], output is
    # [batch_sz, sent_len]
    @staticmethod
    def nll_loss(log_probs, output, **kwargs):
        log_probs_rshp = log_probs.view(-1, log_probs.size(2))
        output_rshp = output.view(-1, output.size(2))
        return F.nll_loss(log_probs_rshp, output_rshp, **kwargs)

class NMTTrainer(NMTModelUser):
    def __init__(self, models, TEXT_SRC, TEXT_TRG, **kwargs):
        super(NMTTrainer, self).__init__(models, TEXT_SRC, TEXT_TRG, **kwargs)

        self.use_attention = kwargs.get('attention', False)

        # TODO: other setup


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
            
        
