# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Naive Bayes
class MultinomialNB:
    def __init__(self):
        self.n_positive = 0
        self.n_negative = 0
        self.p = torch.zeros(len(TEXT.vocab))
        self.q = torch.zeros(len(TEXT.vocab))
        self.r = 0
        
    def get_features(self, case):
        f = np.zeros(len(TEXT.vocab))
        for i in case.data:
            f[i] += 1
        return torch.Tensor(f)
    
    def train(self, train_iter):
        for i in range(len(train_iter)):
            batch = next(iter(train_iter))
            if i % 100 == 0:
                print(i)
            for i in range(batch.text.size()[1]):
                fi = self.get_features(batch.text[:,i])
                if LABEL.vocab.itos[batch.label.data[i]] == "negative":
                    self.n_negative += 1
                    self.p += fi
                elif LABEL.vocab.itos[batch.label.data[i]] == "positive":
                    self.n_positive += 1
                    self.q += fi
        self.r = torch.log((self.p / self.p.sum()) / (self.q / self.q.sum()))
        
    def predict(self, batch_text):
        for k in range(batch_text.size()[1]):
            fk = self.get_features(batch.text[:,k])
            y = self.r * fk + np.log(self.n_positive / self.n_negative)

# Logistic regression; following Torch tutorial
class LogisticRegressionSlow(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(LogisticRegression, self).__init__()
        # TODO: figure out what the <unk> are!!
        self.linear = nn.Linear(len(TEXT.vocab), len(LABEL.vocab))
    
    # Here bow is [N, num-features]
    def forward(self, bow):
        return F.log_softmax(self.linear(bow), dim=1)

class LogisticRegression(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(LogisticRegression2, self).__init__()
        # Embeddings vectors (should be trainable); [V, d]
        # TODO: is default for requires_grad True?
        self.embeddings = nn.EmbeddingBag(len(TEXT.vocab),
                                          len(TEXT.vocab),
                                          mode='sum')
        self.embeddings.weight = nn.Parameter(torch.eye(len(TEXT.vocab)),
                                              requires_grad=False)
        # Linear layer
        self.linear = nn.Linear(len(TEXT.vocab), len(LABEL.vocab))
        
    # Here bow is [len-of-sentence, N] -- it is an integer matrix
    def forward(self, bow):
        bow_features = self.embeddings(bow)
        return F.log_softmax(self.linear(bow_features), dim=1)
    
class CBOW(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(CBOW, self).__init__()
        # Embeddings vectors (should be trainable); [V, d]
        # TODO: is default for requires_grad True?
        self.embeddings = nn.EmbeddingBag(TEXT.vocab.vectors.size()[0],
                                          TEXT.vocab.vectors.size()[1],
                                          mode='sum')
        self.embeddings.weight = nn.Parameter(TEXT.vocab.vectors, requires_grad=True)
        
        # Linear layer
        self.linear = nn.Linear(TEXT.vocab.vectors.size()[1], len(LABEL.vocab))
        
    # Here bow is [len-of-sentence, N] -- it is an integer matrix
    def forward(self, bow):
        bow_features = self.embeddings(bow)
        return F.log_softmax(self.linear(bow_features), dim=1)
    
            
class TextTrainer(object):
    def __init__(self, TEXT, LABEL, model):
        # NLLLoss works with labels, not 1-hot encoding
        self._loss_fn = nn.NLLLoss()
        self._optimizer = optim.SGD(filter(lambda p : p.requires_grad,
                                           model.parameters()), lr=0.1)
        self._TEXT = TEXT
        self._LABEL = LABEL
        self._text_vocab_len = len(self._TEXT.vocab)
        self._model = model

    # TODO: this is horribly slow, can use nn.EmbeddingsBag and put
    # this in LogisticRegression class!
    def get_feature(self, batch):
        # Need transpose so that batch size is first dimension
        return torch.t(batch.text.data).contiguous()
        # size_batch = batch.text.size()[1]
        # features = torch.zeros(size_batch, self._text_vocab_len)
        # for i in range(size_batch):
        #     for j in batch.text[:, i]:
        #         features[i, j.data[0]] += 1
        # return features
    
    def get_label(self, batch):
        return batch.label.data
    
    def make_loss(self, batch):
        bow = autograd.Variable(self.get_feature(batch))
        label = autograd.Variable(self.get_label(batch))
        loss = self._loss_fn(self._model(bow), label)
        return loss
    
    def train(self, train_iter, num_iter=100, skip_iter=10):
        for i in range(num_iter):
            batch = next(iter(train_iter))
            self._model.zero_grad()
            loss = self.make_loss(batch)
            if i % skip_iter == 0:
                print('Iteration %d, loss: %f' % (i, loss))
            loss.backward()
            self._optimizer.step()
            
