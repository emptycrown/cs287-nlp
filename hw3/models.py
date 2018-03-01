import torchtext
from torchtext.vocab import Vectors
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools as it

class EmbeddingsLM(nn.Module):
    def __init__(self, TEXT, dropout=0.0, max_embed_norm=None, word_features=1000):
        super(EmbeddingsLM, self).__init__()
        # Initialize dropout
        self.dropout_prob = dropout
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # V is size of vocab, D is dim of embedding
        self.V = len(TEXT.vocab)
        self.D = word_features
        self.embeddings = nn.Embedding(self.V, self.D, max_norm=max_embed_norm)

class BaseEncoder(EmbeddingsLM):
    def __init__(self, TEXT, hidden_size=1000, num_layers=4,
                 bidirectional=False, **kwargs):
        super(BaseEncoder, self).__init__(TEXT, **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=self.D, hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout_prob, batch_first=True,
                            bidirectional=self.bidirectional)

        
    def forward(self, input_tsr, hidden):
        # [batch_sz, sent_len, D]:
        embedded_tsr = self.embeddings(input_tsr)

        # XXX
        embedded_tsr = self.dropout(embedded_tsr)

        # output is [batch, sent_len, hidden_size * num_directions]
        output, hidden = self.lstm(embedded_tsr, hidden)

        # TODO: this is experimental XXX: should be careful here since
        # the weighted sum of outputs (i.e. context) is already being
        # dropout'ed in the context part of the decoder (but not for
        # attn right now)
        # output = self.dropout(output)
        
        # TODO: perhaps add dropout to output
        return output, hidden

class BaseDecoder(BaseEncoder):
    def __init__(self, TEXT, num_context=1, enc_bidirectional=False, **kwargs):
        super(BaseDecoder, self).__init__(TEXT, **kwargs)
        # V is the size of the vocab, which is what we're predicting
        # (it's also used as input through the embedding)
        self.num_context = num_context
        self.enc_directions = 2 if enc_bidirectional else 1
        # For now assume that encoder and decoder have same hidden size
        blowup = self.num_context * self.num_layers * self.enc_directions + 1
        self.out_linear = nn.Linear(
            blowup * self.hidden_size, self.V)

    # Context is a tuple (h_T, c_T) of hidden and cell states from
    # last time step of encoder
    def forward(self, input_tsr, hidden, context):
        # [batch_sz, sent_len, D] : note that sent_len may be 1 if we
        # feed in each word at a time!
        embedding = self.embeddings(input_tsr)
        embedding = F.relu(embedding)
        output, hidden = self.lstm(embedding, hidden)

        if self.num_context:
            # We get lucky that hidden is stored as (h,c), 
            # so hidden (not cell) first
            context_tsr = torch.cat(context[:self.num_context])
            batch_sz = context_tsr.size(1)
            sent_len = output.size(1)
            # [batch_sz, 1, hidden_size * num_context]
            context_tsr = context_tsr.permute(1,0,2).contiguous().view(batch_sz, 1, -1)
            context_tsr = context_tsr.expand(-1, sent_len, -1)
            # [batch_sz, sent_len, hidden_sz * (num_context + 1)]
            output = torch.cat((output, context_tsr), dim=2)

        # output is now [batch, sent_len, V]:
        output = self.out_linear(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden

class AttnDecoder(BaseEncoder):
    def __init__(self, TEXT, enc_bidirectional=False, tie_weights=False,
                 enc_linear=0, **kwargs):
        super(AttnDecoder, self).__init__(TEXT, **kwargs)
        print('Using final MLP')
        self.enc_directions = 2 if enc_bidirectional else 1
        # XXX
        blowup = self.enc_directions # one for our output, one or two for context
        self.out_linear_dec = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_linear_contxt = nn.Linear(blowup * self.hidden_size, self.hidden_size)
        # self.mlp_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.enc_linear = enc_linear
        if self.enc_linear > 0:
            self.attn_linear = nn.Linear(self.enc_directions * self.enc_linear,
                                         self.hidden_size)

        
        if tie_weights:
            if self.hidden_size != self.D or self.enc_directions != 1:
                raise ValueError('For tied weights, hidden_size must equal num embeddings!')
            self.out_linear_dec.weight = self.embeddings.weight
        
    def forward(self, input_tsr, hidden, enc_output, mask_inds=None):
        # [batch_sz, sent_len, D]:
        embedding = self.embeddings(input_tsr)

        # XXX
        # embedding = F.relu(embedding)
        embedding = self.dropout(embedding)
        
        dec_output, hidden = self.lstm(embedding, hidden)
        
        # Now do attention: enc_output is [batch_sz, sent_len_src, hidden_sz],
        # and dec_output is [batch_sz, sent_len_trg, hidden_sz]
        
        # Normally do linear layer after dropout
        if self.enc_linear > 0:
            enc_output_lin = self.attn_linear(enc_output)
        else:
            enc_output_lin = enc_output

        # enc_output_perm is [batch_sz, hidden_sz, sent_len_src]
        enc_output_perm = enc_output_lin.permute(0, 2, 1)
        
        # should be [batch_sz, sent_len_trg, sent_len_src]
        # Note that decoder hidden state for output pos t is compouted 
        # using hidden state of the last layer (i.e. enc_output) at pos t
        # as opposed to t-1, as in Bahdanau
        dot_products = torch.bmm(dec_output, enc_output_perm)

        # mask_inds is [batch_sz, sent_len_src]
        if not mask_inds is None:
            # np.inf gives nans...
            # Using braodcasting
            mask_inds = autograd.Variable(torch.Tensor([np.inf])) * mask_inds
            # remove nans
            mask_inds[mask_inds != mask_inds] = 0
            dot_products = dot_products - torch.unsqueeze(mask_inds, 1)
        
        # This is the attn distribution, [batch_sz, sent_len_trg, sent_len_src]
        dot_products_sftmx = F.softmax(dot_products, dim=2)

        
        # [batch_sz, sent_len_trg, hidden_sz]
        context = torch.bmm(dot_products_sftmx, enc_output)

        # XXX
        output_1 = self.out_linear_dec(self.dropout(dec_output))
        output_2 = self.out_linear_contxt(self.dropout(context))
        output = output_1 + output_2
        # output = self.mlp_linear(self.dropout(F.tanh(output)))
        output = F.log_softmax(output, dim=2)
        
        # [batch_sz, sent_len_trg, hidden_sz * 2]
        # output = torch.cat((dec_output, context), dim=2)
        # output = self.dropout(output)
        # output = self.out_linear(output)
        # output = F.log_softmax(output, dim=2)
        return output, hidden, dot_products_sftmx      
