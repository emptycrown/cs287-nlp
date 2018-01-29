# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


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
        # Need transpose so that batch size is first dimension; need
        # contiguous because of the transpose
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
    
    def train(self, train_iter, num_iter=100, skip_iter=10, plot=True):
        for i in range(num_iter):
            batch = next(iter(train_iter))
            self._model.zero_grad()
            loss = self.make_loss(batch)
            if i % skip_iter == 0:
                print('Iteration %d, loss: %f' % (i, loss))
            loss.backward()
            self._optimizer.step()
        if plot:
            plt.plot(np.arange(len(self._training_losses)), self._training_losses)
            plt.title("Training loss over time")
            plt.show()
            
    


class TextEvaluator(object):
    def __init__(self):
        self._predictions = []

    def score(self, test_iter, model, predictions_file=""):
        correct = 0
        total = 0
        for i,batch in enumerate(test_iter):
            # Generate feature vector and label
            bow = autograd.Variable(self.get_feature(batch))
            label = autograd.Variable(self.get_label(batch))
            
            # Get predictions
            probs = model(bow)
            _, argmax = probs.max(1)
            if i % 100 == 0:
                print('Iteration %d, predictions:' % (i), list(argmax.data))
            self._predictions += list(argmax.data)

            correct += (self._predictions == batch.label.data).sum()
            total += len(batch.label.data)

        print("Accuracy:", correct/total)
        if predictions_file:
            with open(predictions_file, "w") as f:
                f.write("Id,Cat\n")
                for index,p in enumerate(self._predictions):
                    f.write(str(index) + "," + str(p) + "\n")

            