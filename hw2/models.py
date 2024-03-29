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
        return autograd.Variable(torch.log(ret_arr))
                
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

class EmbeddingsLM(nn.Module):
    def __init__(self, TEXT, **kwargs):
        super(EmbeddingsLM, self).__init__()
        # Initialize dropout
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.5))
        
        # V is size of vocab, D is dim of embedding
        self.V = TEXT.vocab.vectors.size()[0]
        max_embed_norm = kwargs.get('max_embed_norm', 10)
        if kwargs.get('pretrain_embeddings', True):
            self.D = TEXT.vocab.vectors.size()[1]
            self.embeddings = nn.Embedding(self.V, self.D, max_norm=max_embed_norm)
            self.embeddings.weight = nn.Parameter(
                TEXT.vocab.vectors, requires_grad= \
                kwargs.get('train_embeddings', True))
        else:
            self.D = kwargs.get('word_features', 100)
            self.embeddings = nn.Embedding(self.V, self.D, max_norm=max_embed_norm)
        

class NNLM(EmbeddingsLM):
    def __init__(self, TEXT, **kwargs):
        # sets up self.embeddings, self.D, self.V, self.dropout
        super(NNLM, self).__init__(TEXT, **kwargs)

        # Save parameters:
        self.activation = kwargs.get('activation', F.tanh)
                
        in_channels = 1
        out_channels = kwargs.get('hidden_size', 100)
        self.kernel_sizes_inner = [kwargs.get('kern_size_inner', 5)] 
        self.kernel_size_direct = kwargs.get('kern_size_direct', -1)

        # List of convolutional layers
        self.convs_inner = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, (K, self.D),
                       padding=(K, 0)) for K in self.kernel_sizes_inner])
        if self.kernel_size_direct > 0:
            # Bias is already in self.linear, so don't put another here
            self.conv_direct = nn.Conv2d(
                in_channels, self.V, (self.kernel_size_direct, self.D),
                padding=(self.kernel_size_direct,0), bias=False)

        
        self.linear = nn.Linear(len(self.kernel_sizes_inner) * out_channels,
                                self.V)
    
    # x is [batch_sz, sent_len]: words are encoded as integers (indices)
    def forward(self, x):
        x = self.embeddings(x) # [btch_sz, sent_len, D]
        x = x.unsqueeze(1) # [btch_sz, in_channels, sent_len, D]
        # [btch_sz, out_channels, sent_len] * len(kerns)
        x = [self.activation(conv(x)).squeeze(3)\
             [:,:,:-(self.kernel_sizes_inner[i]+1)] for \
             i,conv in enumerate(self.convs_inner)]
        # [btch_sz, out_channels * len(kerns), sent_len]
        x = torch.cat(x, 1)
        # [btch_sz, sent_len, out_channels * len(kerns)]
        x = x.permute(0, 2, 1)
        
        x = self.dropout(x) # Bengio et al. doesn't mention dropout 
        # (it hadn't been 'discovered')
        
        # [btch_sz, sent_len, V]
        x = self.linear(x) # has a bias term
        
        if self.kernel_size_direct > 0:
            # [btch_sz, V, sent_len]
            y = self.conv_direct(x)[:,:,:-(self.kernel_size_direct+1)]
            # [btch_sz, sent_len, V]
            y = y.permute(0, 2, 1)
            x = x + y # '+' should be overloaded
            
        return F.log_softmax(x, dim=2)        

class LSTMLM2(EmbeddingsLM):
    def __init__(self, TEXT, **kwargs):
        # sets up self.D, self.V, self.embeddings, self.dropout
        super(LSTMLM2, self).__init__(TEXT, **kwargs)
        
        # Save parameters:
        self.hidden_dim = kwargs.get('hidden_size', 650)
        self.num_layers = kwargs.get('num_layers', 2)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # TODO: Make sure LSTM does dropout the right way on the inner parameters
        self.lstm = nn.LSTM(input_size=self.D, hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=kwargs.get('dropout', 0.5),
                            batch_first=True)
        
        # The linear layer that maps from hidden state space to label space
        self.linear = nn.Linear(self.hidden_dim, self.V)

        if kwargs.get('tie_weights', False):
            if self.hidden_dim != self.D:
                raise ValueError('For tied weights, hidden dim must be equal to num embeddings!')
            self.linear.weight = self.embeddings.weight

    # hidden should be [batch_sz, num_layers, hidden_dim]
    def forward(self, x, hidden):
        # sent_len = x.size(1)
        # btch_sz = x.size(0)

        x = self.embeddings(x) # [btch_sz, sent_len, D]

        # Put dropout before the LSTM as in the paper
        x = self.dropout(x)
        # hidden_out is ([batch_sz, num_layers, hidden_dim]) * 2
        lstm_out, hidden_out = self.lstm(x, hidden)

        # lstm_out is [batch_sz, sent_len, hidden]
        lstm_out = self.dropout(lstm_out)
        # pred is [batch_sz, sent_len, V]
        pred = self.linear(lstm_out)

        return F.log_softmax(pred, dim=2), hidden_out
    

# OLD VERSION
class LSTMLM(nn.Module):
    def __init__(self, TEXT, **kwargs):
        super(LSTMLM, self).__init__()
        
        # Save parameters:
        self.hidden_dim = kwargs.get('hidden_dim', 650)
        self.btch_sz = kwargs.get('btch_sz', 10)
 
        # V is size of vocab, D is dim of embedding
        V = TEXT.vocab.vectors.size()[0]
        D = TEXT.vocab.vectors.size()[1]
        self.embeddings = nn.Embedding(V, D)
        self.embeddings.weight = nn.Parameter(
            TEXT.vocab.vectors, requires_grad= \
            kwargs.get('train_embeddings', True))
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(D, self.hidden_dim)
        
        # The linear layer that maps from hidden state space to label space
        self.linear = nn.Linear(self.hidden_dim, V)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, self.btch_sz, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, self.btch_sz, self.hidden_dim).cuda()))
        
    def forward(self, x):
        sent_len = x.size(1)
        btch_sz = x.size(0)
        x = self.embeddings(x) # [btch_sz, sent_len, D]
        lstm_out, self.hidden = self.lstm(
            x.view(sent_len, btch_sz, -1), self.hidden)
        pred = self.linear(lstm_out.view(sent_len, btch_sz, -1))
        return F.log_softmax(pred, dim=2)
