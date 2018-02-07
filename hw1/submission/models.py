# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MultinomialNB(nn.Module):
    def __init__(self, TEXT, LABEL, bag_or_set='bag'):
        super(MultinomialNB, self).__init__()
        self._bag_or_set = bag_or_set
        self._TEXT = TEXT
        self._LABEL = LABEL
        self._text_vocab_len = len(self._TEXT.vocab)        
        self.n_positive = 0
        self.n_negative = 0
        # Smoothing para is 1 for all features
        self.p = torch.ones(self._text_vocab_len)
        self.q = torch.ones(self._text_vocab_len)
        self.r = None
        self.index_pos = LABEL.vocab.itos.index('positive')
        self.index_neg = LABEL.vocab.itos.index('negative')

    # could use EmbeddingsBag, but there's not a huge difference in
    # performance
    def get_features(self, batch):
        size_batch = batch.size()[0]
        features = torch.zeros(size_batch, self._text_vocab_len)
        for i in range(size_batch):
            for j in batch[i, :]:
                if self._bag_or_set == 'bag':
                    features[i, j.data[0]] += 1
                elif self._bag_or_set == 'set':
                    features[i, j.data[0]] = 1
                else:
                    raise ValueError('Invalid value for bag_or_set: %s' % \
                                     self._bag_or_set)
        return features
        # return torch.Tensor(features)

    def train(self, train_iter):
        # There's probably a better way to do this
        num_iter = len(train_iter)
        train_iter = iter(train_iter)
        for i in range(num_iter):
            batch = next(train_iter)
            if i % 100 == 0:
                print(i)
            # Should be [N, num-features]
            features = self.get_features(torch.t(batch.text).contiguous())

            # Using broadcasting
            inds_pos = torch.nonzero(batch.label.data == self.index_pos)
            inds_neg = torch.nonzero(batch.label.data == self.index_neg)


            if inds_pos.size():
                self.n_positive += inds_pos.size()[0]
                self.p = torch.add(self.p, torch.sum(features[inds_pos, :], dim=0))                
            if inds_neg.size():
                self.n_negative += inds_neg.size()[0]
                self.q = torch.add(self.q, torch.sum(features[inds_neg, :], dim=0))

            # print(features)
            # print(inds_neg, inds_pos)
            # print(self.p.size(), torch.sum(features, dim=0).size())

        self.r = torch.log((self.p / self.p.sum()) / (self.q / self.q.sum()))
        
    def forward(self, batch):
        # for k in range(batch_text.size()[1]):
        features = self.get_features(batch)
        # Using broadcasting
        return torch.matmul(features, torch.squeeze(self.r)) + \
            np.log(self.n_positive / self.n_negative)


# Logistic regression; following Torch tutorial
class LogisticRegressionSlow(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(LogisticRegression, self).__init__()
        # TODO: figure out what the <unk> are!!
        self.linear = nn.Linear(len(TEXT.vocab), len(LABEL.vocab))
    
    # Here bow is [N, num-features]
    def forward(self, bow):
        return F.log_softmax(self.linear(bow), dim=1)

# This ends up not being much faster than LogisticRegressionSlow
# (perhaps even a bit slower...), but it's more elegant...
class LogisticRegression(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(LogisticRegression, self).__init__()
        # Embeddings vectors (should be trainable); [V, d]
        # TODO: is default for requires_grad True?
        self.embeddings = nn.EmbeddingBag(len(TEXT.vocab),
                                          len(TEXT.vocab),
                                          mode='sum')
        self.embeddings.weight = nn.Parameter(torch.eye(len(TEXT.vocab)),
                                              requires_grad=False)
        self.linear = nn.Linear(len(TEXT.vocab), len(LABEL.vocab))
    
    # Here bow is [len-of-sentence, N] -- it is an integer matrix
    def forward(self, bow):
        bow_features = self.embeddings(bow)
        return F.log_softmax(self.linear(bow_features.float()), dim=1)
    
class CBOW(nn.Module):
    def __init__(self, TEXT, LABEL, dynamic=True):
        super(CBOW, self).__init__()
        # Embeddings vectors (should be trainable); [V, d]
        # TODO: is default for requires_grad True?
        self.embeddings = nn.EmbeddingBag(TEXT.vocab.vectors.size()[0],
                                          TEXT.vocab.vectors.size()[1],
                                          mode='sum')
        self.embeddings.weight = nn.Parameter(TEXT.vocab.vectors,
                                              requires_grad=dynamic)
        
        # Linear layer
        self.linear = nn.Linear(TEXT.vocab.vectors.size()[1], len(LABEL.vocab))
        
    # Here bow is [len-of-sentence, N] -- it is an integer matrix
    def forward(self, bow):
        bow_features = self.embeddings(bow)
        return F.log_softmax(self.linear(bow_features), dim=1)
    
class CNN(nn.Module):
    def __init__(self, TEXT, LABEL, do_binary=False, activation=F.relu,
                 train_embeddings=True):
        super(CNN, self).__init__()

        # Save parameters:
        self.activation = activation
        
        N = TEXT.vocab.vectors.size()[0]
        D = TEXT.vocab.vectors.size()[1]
        C = 1 if do_binary else len(LABEL.vocab)
        if do_binary:
            self.index_pos = 1
            self.index_neg = 0
        self.do_binary = do_binary
        in_channels = 1
        out_channels = 100
        kernel_sizes = [3, 4, 5] 
        
        self.embeddings = nn.Embedding(N, D)
        self.embeddings.weight = nn.Parameter(TEXT.vocab.vectors, requires_grad=train_embeddings)
        
        # List of convolutional layers
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channels,
                                               out_channels,
                                               (K, D),
                                               padding=(K-1, 0)) \
                                     for K in kernel_sizes])

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(kernel_sizes)*out_channels, C)

    def forward(self, x):
        x = self.embeddings(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, in_channels, W, D)
        x = [self.activation(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, out_channels, W)]*len(kernel_sizes)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, out_channels)]*len(kernel_sizes)
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(kernet_sizes)*out_channels)
        if not self.do_binary:
            return F.log_softmax(self.fc1(x), dim=1)  # (N, C)
        else:
            return torch.squeeze(self.fc1(x), dim=1)
