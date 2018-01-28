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
class LogisticRegression(nn.Module):
    def __init__(self, TEXT, LABEL):
        super(LogisticRegression, self).__init__()
        # TODO: figure out what the <unk> are!!
        self.tlinear = nn.Linear(len(TEXT.vocab), len(LABEL.vocab))
    
    # Here bow is [N, num-features]
    def forward(self, bow):
        return F.log_softmax(self.linear(bow), dim=1)

            
class TextTrainer(object):
    def __init__(self, TEXT, LABEL):
        # NLLLoss works with labels, not 1-hot encoding
        self._loss_fn = nn.NLLLoss()
        self._optimizer = optim.SGD(model.parameters(), lr=0.1)
        self._TEXT = TEXT
        self._LABEL = LABEL
        self._text_vocab_len = len(self._TEXT.vocab)
        
    def get_feature(self, batch):
        size_batch = batch.text.size()[1]
        features = torch.zeros(size_batch, self._text_vocab_len)
        # TODO: find a better way to do this!
        for i in range(size_batch):
            for j in batch.text[:, i]:
                features[i, j.data[0]] += 1
        return features
    
    def get_label(self, batch):
        return batch.label.data
    
    def make_loss(self, batch):
        bow = autograd.Variable(self.get_feature(batch))
        label = autograd.Variable(self.get_label(batch))
        loss = self._loss_fn(model(bow), label)
        return loss
    
    def train(self, train_iter, model):
        for i in range(10):
            batch = next(iter(train_iter))
            model.zero_grad()
            loss = self.make_loss(batch)
            print('Iteration %d, loss: %f' % (i, loss))
            loss.backward()
            self._optimizer.step()
            
