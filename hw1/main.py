# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import *

def main():
    # Our input $x$
    TEXT = torchtext.data.Field()
    
    # Our labels $y$
    LABEL = torchtext.data.Field(sequential=False)


    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')

    # Build vocab
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=-1)

    if args.model == 'nb':
        model = MultinomialNB()
        model.train(train_iter)
    elif args.model == 'lr':
        model = LogisticRegression()
        trainer = TextTrainer(TEXT, LABEL)
        trainer.train(train_iter, model)
    elif args.model == 'cbow':
        raise NotImplementedError('CBOW')
    elif args.model == 'cnn':
        raise NotImplementedError('CNN')
    else:
        raise ValueError('Invalid model %s' % args.model)


    # TODO: test

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    add_parse(parser)
    args = parser.parse_args()
    main(args)

def add_parse(parser):
    parser.add_argument('--model', action='store', default='nb')
    
