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
             'nnlm' : NNLM,
             'lstmlm' : LSTMLM2}

RNN_NAMES = ['lstmlm']

OPT_NAMES = {'sgd' : optim.SGD,
             'adam' : optim.Adam,
             'adagrad' : optim.Adagrad,
             'adadelta' : optim.Adadelta}

ACT_NAMES = {'tanh' : F.tanh,
             'relu' : F.relu,
             'hardtanh' : F.hardtanh}

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--network', default='nnlm')
    parser.add_argument('--batch_sz', type=int, default=10)
    parser.add_argument('--bptt_len', type=int, default=32)
    parser.add_argument('--early_stop', action='store_true', default=False)

    # Arguments for LangTrainer:
    parser.add_argument('--t_lrn_rate', type=float, default=0.1)
    parser.add_argument('--t_lrn_decay', default='none')
    parser.add_argument('--t_lrn_decay_rate', type=float, default=1.0)
    parser.add_argument('--t_clip_norm', type=int, default=-1)
    parser.add_argument('--t_optimizer', default='sgd')
    parser.add_argument('--t_retain_graph', action='store_true',
                        default=False)

    # ARguments for model:
    parser.add_argument('--m_pretrain_embeddings', action='store_true',
                        default=False)
    parser.add_argument('--m_word_features', type=int, default=100)
    parser.add_argument('--m_hidden_size', type=int, default=100)
    parser.add_argument('--m_kern_size_inner', type=int, default=5)
    parser.add_argument('--m_kern_size_direct', type=int, default=-1)
    parser.add_argument('--m_dropout', type=float, default=0.5)
    parser.add_argument('--m_num_layers', type=int, default=1)
    
    # Process of training args:
    parser.add_argument('--tt_num_iter', type=int, default=100)
    parser.add_argument('--tt_skip_iter', type=int, default=1)
    parser.add_argument('--tt_produce_predictions', action='store_true', 
                        default=False)

    args = parser.parse_args()
    return args

def prepare_special_args(args_dict, key_full, key_match):
    if key_match == 'optimizer':
        return OPT_NAMES[args_dict[key_full]]
    elif key_match == 'activation':
        return ACT_NAMES[args_dict[key_full]]
    else:
        return args_dict[key_full]

def prepare_kwargs(args, root):
    ret_dict = dict()
    args_dict = vars(args)
    for key in args_dict:
        root_len = len(root) + 1
        if key[:root_len] == root + '_':
            print('Argument: %s, Value: %s' % (key, args_dict[key]))
            ret_dict[key[root_len:]] = prepare_special_args(
                args_dict, key, key[root_len:])
    return ret_dict

# train_val_test is a tuple of those datasets
def train_network(net_name, args, TEXT, train_val_test):
    model = NET_NAMES[net_name](TEXT, **prepare_kwargs(args, 'm'))
    trainer = LangTrainer(TEXT, model, use_hidden=(net_name in RNN_NAMES),
                          **prepare_kwargs(args, 't'))

    _, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
        train_val_test, batch_size=args.batch_sz, device=-1,
        bptt_len=args.bptt_len, repeat=False)
    train_iter, _, _ = torchtext.data.BPTTIterator.splits(
        train_val_test, batch_size=args.batch_sz, device=-1,
        bptt_len=args.bptt_len, repeat=False) # TODO: shuffle?
    if args.early_stop:
        le = LangEvaluator(model, TEXT, use_hidden=(net_name in RNN_NAMES))
        return trainer.train(train_iter, le=le, val_iter=val_iter, test_iter=test_iter,
                      retain_graph=(args.t_retain_graph),
                      **prepare_kwargs(args, 'tt'))
    else:
        return trainer.train(train_iter, **prepare_kwargs(args, 'tt'))


def main(args):
    # Basic setup
    # Our input $x$
    TEXT = torchtext.data.Field()

    # Data distributed with the assignment
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
        path=".", 
        train="train.txt", validation="valid.txt", test="input.txt", text_field=TEXT)

    TEXT.build_vocab(train)
    if args.debug:
        TEXT.build_vocab(train, max_size=1000)

    # train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    #     (train, val, test), batch_size=10, device=-1, bptt_len=32, repeat=False)

    # Build the vocabulary with word embeddings
    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

    return train_network(args.network, args, TEXT, (train, val, test))

        
if __name__ == '__main__':
    args = parse_input()
    main(args)
    print('SUCCESS')
