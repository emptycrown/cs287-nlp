import torchtext
from torchtext.vocab import Vectors
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools as it

class Trigram(nn.Module):
    def __init__(self, TEXT, **kwargs):
        super(Trigram, self).__init__()
        self._TEXT = TEXT
        self._text_vocab_len = len(TEXT.vocab)
        
        # Use dictionaries since we don't want to have to 
        # store the vast majority of bi/tri gram counts 
        # which are 0.
        self.cnts = [dict(), dict(), dict()]
        
    def set_alpha(self, *args):
        self.alphas = list(args)
        if len(self.alphas) < 3:
            assert len(self.alphas) == 2
            self.alphas.append(1 - sum(self.alphas))
            
    def normalize_cnts(self):
        for n in range(3):
            for k in self.cnts[n]:
                self.cnts[n][k] = self.cnts[n][k] / torch.sum(self.cnts[n][k])
        
    def train_counts(self, train_iter, num_iter=None):
        if num_iter is None:
            num_iter = len(train_iter)
        train_iter = iter(train_iter)
        for i in range(num_iter):
            batch = next(train_iter)
            if i % 1000 == 0:
                print('Iteration %d' % i)
            self.update_trigram_cnts(torch.t(batch.text).data)
        self.normalize_cnts()
            
    # Returns a torch tensor of size [size_batch, sentence_len,
    # size_vocab]; this returns the probability vectors for each of
    # the words This is quite slow, but the trigram model isn't a
    # bottleneck so there's no point in making it faster
    def forward(self, batch):
        ret_arr = torch.zeros(batch.size()[0], batch.size()[1], 
                              self._text_vocab_len)
        for i in range(batch.size()[0]):
            for n in range(0,3):
                for j in range(batch.size()[1]):
                    key = () if max(0, j-n) == j else \
                          tuple(batch.data[i, max(0, j-n):j])
                    if key in self.cnts[n]:
                        ret_arr[i,j,:] += self.alphas[n] * self.cnts[n][key]
                    else:
                        # This is equivalent to adding NB-type
                        # smoothing, but more efficient memory-wise
                        ret_arr[i,j,:] += self.alphas[n] * torch.ones(
                            self._text_vocab_len) / self._text_vocab_len
        # Use log probabilities to make this work with LangEvaluator
        # case_sums = torch.sum(ret_arr, dim=2, keepdim=True)
        # print(torch.sum(case_sums, dim=2))
        # Using broadcasting
        return torch.log(ret_arr)
                
    # Batch is an torch tensor of size [batch_size, bptt_len]
    def update_trigram_cnts(self, batch):
        # We don't glue rows together since they may be shuffled (this
        # is all kind of silly since ideally we'd just do this in one
        # big 'sentence', but perhaps we want a 'fair comparison'...)
        for j in range(batch.shape[0]):
            for n in range(0,3):
                for k in range(batch.shape[1] - n):
                    dict_key = () if k == k+n else tuple(batch[j, k:k+n])
                    if not dict_key in self.cnts[n]:
                        # We never want to return 0 probability (or
                        # else PPL = infty), so make sure we've got
                        # mass everywhere
                        self.cnts[n][dict_key] = torch.zeros(self._text_vocab_len) if \
                                                 n > 0 else torch.ones(self._text_vocab_len)
                    # Here's where we increment the ocunt
                    self.cnts[n][dict_key][batch[j, k+n]] += 1    
