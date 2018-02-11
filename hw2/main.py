# Text text processing library
import torchtext
from torchtext.vocab import Vectors
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools as it
from models import *
from helpers import *
import argparse

NET_NAMES = {'trigram' : Trigram,
             'nnlm' : NNLM}

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--networks', nargs='+',
                        default=['nnlm'])
    parser.add_argument('--batch_sz', type=int, default=10)
    parser.add_argument('--bptt_len', type=int, default=32)
    args = parser.parse_args()
    return args

def prepare_kwargs(args, root):
    ret_dict = dict()
    args_dict = vars(args)
    for key in args_dict:
        if key[:2] == root + '_':
            ret_dict[key[2:]] = args_dict[key]
    return ret_dict

# train_val_test is a tuple of those datasets
def train_network(net_name, args, TEXT, train_val_test):
    model = NET_NAMES[net_name](TEXT, **prepare_kwargs(args, 'm'))
    trainer = LangTrainer(TEXT, model, **prepare_kwargs(args, 't'))

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        train_val_test, batch_size=args.batch_sz, device=-1,
        bptt_len=args.bptt_len, repeat=False)

    trainer.train(train_iter)


def main(args):
    # Basic setup
    # Our input $x$
    TEXT = torchtext.data.Field()

    # Data distributed with the assignment
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path=".", 
        train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

    TEXT.build_vocab(train)
    if args.debug:
        TEXT.build_vocab(train, max_size=1000)

    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        (train, val, test), batch_size=10, device=-1, bptt_len=32, repeat=False)

    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    for net_name in args.networks:
        train_network(net_name, args, TEXT, (train, val, test))

        
if __name__ == '__main__':
    args = parse_input()
    main(args)
    print('SUCCESS')
