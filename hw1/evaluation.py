# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from submission.helpers import TextTrainer

def save_test(model, test):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        _, argmax = probs.max(1)
        upload += list(argmax.data)

    with open("predictions.txt", "w") as f:
        for u in upload:
            f.write(str(u) + "\n")

def model_eval(model, test, test_iter=None, do_binary=False):
    if test_iter is None:
        test_iter = torchtext.data.BucketIterator(test, train=False,
                                                  batch_size=10,
                                                  device=-1)
    cnt_correct = 0
    cnt_total = 0
    for batch in test_iter:
        classes = get_predictions(model, batch)
        cnt_total += batch.text.size()[1]
        # print(batch.label == argmax, (batch.label == argmax).sum().data[0])
        cnt_correct += (classes == TextTrainer.get_label(batch, do_binary,
                                                         type_return=torch.LongTensor)).sum()
    return (cnt_correct, cnt_total)

def get_predictions(model, batch):
    probs = model(torch.t(batch.text).contiguous())
    
    if len(probs.size()) == 1 or (len(probs.size()) == 2 \
                                  and probs.size()[1] == 1):
        signs = torch.sign(probs).type(torch.LongTensor)
        assert(hasattr(model, 'index_pos'))
        classes = (signs + 1) * (model.index_pos - model.index_neg) / 2 + \
                  model.index_neg
        # hack here; should be more carefuly about this
        if hasattr(classes, 'data'):
            classes = classes.data
        # print(classes, probs)
    else:
        _, argmax = probs.max(1)
        classes = argmax.data
    return classes

def model_save_predictions(model, test_iter, predictions_file='predictions.txt'):
    predictions_arr = list()
    for i,batch in enumerate(test_iter):            
        # Get predictions
        predictions = get_predictions(model, batch)
        # if i % 100 == 0:
        #     print('Iteration %d, predictions:' % (i), list(predictions))
        predictions_arr += list(predictions)

        if predictions_file:
            with open(predictions_file, 'w') as f:
                f.write('Id,Cat\n')
                for index,p in enumerate(predictions_arr):
                    f.write(str(index) + ',' + str(p) + '\n')
