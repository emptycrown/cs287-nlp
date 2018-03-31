import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import time
import matplotlib
# hack to make matplotlib work
if torch.cuda.is_available():
    matplotlib.use('Agg')

from models import *
from helpers import *
import argparse

# New stuff.
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

OPT_NAMES = {'sgd' : optim.SGD,
             'adam' : optim.Adam,
             'adagrad' : optim.Adagrad,
             'adadelta' : optim.Adadelta}

GAN_CLASSES = {'ganmlp' : [MLPDiscriminator, MLPGenerator],
               'ganconv' : [MLPDiscriminator, DeconvGenerator]}

def parse_input(input_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='vaestd')
    parser.add_argument('--batch_sz', type=int, default=100)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--load_model_fn', default=None)
    parser.add_argument('--make_plots', action='store_true', default=False)

    # Arguments for Trainer
    # First learning rate is D, second is G
    parser.add_argument('--t_lrn_rate', type=float, nargs='+', default=0.1)
    parser.add_argument('--t_optimizer', default='sgd')

    # Arguments for model:
    parser.add_argument('--m_latent_dim', type=int, default=10)
    parser.add_argument('--m_hidden_dim', type=int, default=200)

    # Process of training args:
    parser.add_argument('--tt_num_epochs', type=int, default=100)
    parser.add_argument('--tt_skip_epochs', type=int, default=1)
    parser.add_argument('--tt_save_model_fn', default=None)
    parser.add_argument('--tt_gan_k', type=int, default=1)
    
    args = parser.parse_args(args=input_str)
    return args


def prepare_special_args(args_dict, key_full, key_match):
    if key_match == 'optimizer':
        return OPT_NAMES[args_dict[key_full]]
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

    # Special cases:
    if args.network in VAE_MODES and 'gan_k' in ret_dict:
        del ret_dict['gan_k']
    return ret_dict

def load_enc_dec(classes_models, svd_models, args):
    cuda_status = args.cuda and torch.cuda.is_available()
    for i in range(len(classes_models)):
        set_parameters(classes_models[i], svd_models[i],
                       cuda_status)

def run_vae(args, train_loader, val_loader, test_loader, svd_models=None):
    # Do case work
    if args.network == 'vaestd':
        encoder_name = MLPEncoder
        decoder_name = MLPDecoder
        vae_name = NormalVAE
    elif args.network == 'vaeiaf':
        encoder_name = LinearIAFMLPEncoder
        decoder_name = MLPDecoder
        vae_name = IAFNormalVAE
    else:
        raise ValueError('Invalid network %s' % args.network)
        
    encoder_mlp = encoder_name(**prepare_kwargs(args, 'm'))
    decoder_mlp = decoder_name(**prepare_kwargs(args, 'm'))
    if not svd_models is None:
        load_enc_dec([encoder_mlp, decoder_mlp], svd_models, args)

    vae = vae_name(encoder_mlp, decoder_mlp)    
    model_list = [encoder_mlp, decoder_mlp, vae]
    lm_evaluator = LatentModelEvaluator(model_list, batch_sz=args.batch_sz, mode=args.network,
                                        cuda=args.cuda)
    if svd_models is None:
        lm_trainer = VAELatentModelTrainer(model_list, **prepare_kwargs(args, 't'),
                                           batch_sz=args.batch_sz, mode=args.network,
                                           cuda=args.cuda)
        lm_trainer.train(train_loader, le=lm_evaluator, val_loader=val_loader,
                         **prepare_kwargs(args, 'tt'))
    test_loss, test_kl = lm_evaluator.evaluate(test_loader)
    print('Test results: loss: %f, KL : %f' % (test_loss, test_kl))

    if args.make_plots:
        # TODO: The num_batch=10 is temporary!
        lm_evaluator.make_vae_plots(test_loader, num_batch=10)

def run_gan(args, train_loader, val_loader, test_loader, svd_models=None):
    gan_classes = GAN_CLASSES[args.network]
    gen_mlp = gan_classes[1](**prepare_kwargs(args, 'm'))
    disc_mlp = gan_classes[0](**prepare_kwargs(args, 'm'))
    if not svd_models is None:
        load_enc_dec([disc_mlp, gen_mlp], svd_models, args)

    model_list = [disc_mlp, gen_mlp]
    lm_evaluator = LatentModelEvaluator(model_list, batch_sz=args.batch_sz,
                                        mode=args.network, cuda=args.cuda)
    if svd_models is None:
        lm_trainer = GANLatentModelTrainer(model_list, **prepare_kwargs(args, 't'),
                                           batch_sz=args.batch_sz, mode=args.network,
                                           cuda=args.cuda)
        lm_trainer.train(train_loader, le=lm_evaluator, val_loader=val_loader,
                         **prepare_kwargs(args, 'tt'))
    test_loss_disc, test_loss_g = lm_evaluator.evaluate(test_loader)
    print('Test results: Disc acc real: %f, Disc acc fake: %f, GAN obj: %f' % \
          (test_loss_disc[0] * 2, test_loss_disc[1] * 2, test_loss_g))

    if args.make_plots:
        lm_evaluator.make_gan_plots()
    
def main(args):
    train_dataset = datasets.MNIST(root='./data/',
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)
    test_dataset = datasets.MNIST(root='./data/',
                               train=False, 
                               transform=transforms.ToTensor())
    torch.manual_seed(3435)
    if args.network in VAE_MODES:
        train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
        test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    else:
        train_img = torch.stack([d[0] for d in train_dataset])
        test_img = torch.stack([d[0] for d in test_dataset])
    train_img = torch.squeeze(train_img)
    test_img = torch.squeeze(test_img)
    train_label = torch.LongTensor([d[1] for d in train_dataset])
    test_label = torch.LongTensor([d[1] for d in test_dataset])

    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()
    train_img = train_img[:-10000]
    train_label = train_label[:-10000]

    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)
    test = torch.utils.data.TensorDataset(test_img, test_label)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_sz, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_sz, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_sz, shuffle=True)

    svd_models = None
    if not args.load_model_fn is None:
        svd_models = load_checkpoint('saved_models/' + args.load_model_fn)

    if args.network in VAE_MODES:
        run_vae(args, train_loader, val_loader, test_loader,
                svd_models=svd_models)
    elif args.network in GAN_MODES:
        run_gan(args, train_loader, val_loader, test_loader,
                svd_models=svd_models)
    else:
        raise ValueError('Invalid network argument: got %s, expected vae or gan' %
                         args.network)

if __name__ == '__main__':
    args = parse_input()
    main(args)
    print('SUCCESS')
