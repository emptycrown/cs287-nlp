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
    def __init__(self, TEXT, LABEL, model, optimizer=optim.SGD):
        # NLLLoss works with labels, not 1-hot encoding
        self._loss_fn = nn.NLLLoss()
        self._optimizer = optimizer(filter(lambda p : p.requires_grad,
                                           model.parameters()), lr=0.1)
        self._TEXT = TEXT
        self._LABEL = LABEL
        self._text_vocab_len = len(self._TEXT.vocab)
        self._model = model
        
        # For review and assessment
        self._training_losses = []

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
    
    def train(self, train_iter, num_iter=1000, skip_iter=100, plot=True):
        for i in range(num_iter):
            batch = next(iter(train_iter))
            self._model.zero_grad()
            loss = self.make_loss(batch)
            
            loss_val = loss.data.numpy()[0] 
            self._training_losses.append(loss_val)
            if i % skip_iter == 0:
                print('Iteration %d, loss: %f' % (i, loss_val))
            loss.backward()
            self._optimizer.step()
        if plot:
            plt.plot(np.arange(len(self._training_losses)), self._training_losses)
            plt.title("Training loss over time")
            plt.show()

class TextEvaluator(object):
    def __init__(self, model):
        self._predictions = []
        self._model = model

    def score(self, test_iter, predictions_file=""):
        correct = 0
        total = 0
        for i,batch in enumerate(test_iter):
            # Get predictions
            probs = self._model(torch.t(batch.text).contiguous())
            if len(probs.size()) == 1 or (len(probs.size()) == 2 \
                                          and probs.size()[1] == 1):
                signs = torch.sign(probs).type(torch.LongTensor)
                classes = (signs + 1) * (self._model.index_pos - self._model.index_neg) / 2 + \
                          self._model.index_neg
                # print(classes, probs)
            else:
                _, argmax = probs.max(1)
                classes = argmax.data
            if i % 100 == 0:
                print('Iteration %d, predictions:' % (i), list(classes))
            self._predictions += list(classes)

            correct += (classes == batch.label.data).sum()
            total += len(batch.label.data)

        print("Accuracy:", correct, total, correct/total)
        if predictions_file:
            with open(predictions_file, "w") as f:
                f.write("Id,Cat\n")
                for index,p in enumerate(self._predictions):
                    f.write(str(index) + "," + str(p) + "\n")

            

            
