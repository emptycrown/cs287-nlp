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
    def __init__(self, TEXT, **kwargs):
        super(EmbeddingsLM, self).__init__()
        # Initialize dropout
        self.dropout_prob = kwargs.get('dropout', 0.0)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # V is size of vocab, D is dim of embedding
        self.V = TEXT.vocab.vectors.size()[0]
        max_embed_norm = kwargs.get('max_embed_norm', None)
        self.D = kwargs.get('word_features', 1000)
        self.embeddings = nn.Embedding(self.V, self.D, max_norm=max_embed_norm)

class BaseEncoder(EmbeddingsLM):
    def __init__(self, TEXT, **kwargs):
        super(BaseEncoder, self).__init__(TEXT, **kwargs)
        self.hidden_size = kwargs.get('hidden_size', 1000)
        self.num_layers = kwargs.get('num_layers', 4)
        self.lstm = nn.LSTM(input_size=self.D, hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=self.dropout_prob, batch_first=True)
        
    def forward(self, input_tsr, hidden):
        # [batch_sz, sent_len, D]:
        embedded_tsr = self.embedding(input_vec)

        # output is [batch, sent_len, hidden_size]
        output, hidden = self.lstm(embedded_tsr, hidden)
        
        # TODO: perhaps add dropout to output
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class BaseDecoder(BaseEncoder):
    def __init__(self, TEXT, **kwargs):
        super(BaseDecoder, self).__init__(TEXT, **kwargs)
        # V is the size of the vocab, which is what we're predicting
        # (it's also used as input through the embedding)
        self.use_cell = kwargs.get('use_cell', False)
        # For now assume that encoder and decoder have same hidden size
        blowup = 3 if self.use_cell else 2
        self.out_linear = nn.Linear(blowup * self.hidden_size, self.V)

    # Context is a tuple (h_T, c_T) of hidden and cell states from
    # last time step of encoder
    def forward(self, input_tsr, hidden, context):
        # [batch_sz, sent_len, D] : note that sent_len may be 1 if we
        # feed in each word at a time!
        embedding = self.embedding(input_tsr)
        embedding = F.relu(embedding)
        output, hidden = self.lstm(embedding, hidden)

        if self.use_cell:
            output = torch.cat(context + (output,), dim=2)
        else:
            output = torch.cat((context[0], output), dim=2)

        # output is now [batch, sent_len, V]:
        output = self.out_linear(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
