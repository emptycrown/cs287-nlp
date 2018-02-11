import itertools as it
import main
import argparse

def grid_srch():
    base_dict = {'tt_num_iter' : 100000,
                 'early_stop' : True,
                 'batch_sz' : 1,
                 't_lrn_decay' : 'invlin',
                 't_optimizer' : 'adam',
                 'm_kern_size_inner' : 4}

    lr_bases = [0.0005, 0.001, 0.002]
    lr_decays = [0.05, 0.1, 0.2]
    bptt_lens = [1000, 2000, 4000]
    hidden_szs = [100]
    word_ftrs = [100]

    cnt = 0
    best_valids = list()
    for (lr_base, lr_decay, bptt, hidden, word) in \
        it.product(lr_bases, lr_decays, bptt_lens,
                   hidden_szs, word_ftrs):
        cnt += 1
        print('Setting %d of parameters' % cnt)
        param_dict = {'t_lrn_rate' : lr_base,
                      't_lrn_decay_rate' : lr_decay,
                      'bptt_len' : bptt,
                      'm_hidden_size' : hidden,
                      'm_word_features' : word}
        param_dict.update(base_dict)
        args = argparse.Namespace(**param_dict)
        print(args)
        args_def_dict = vars(main.parse_input())
        args_def_dict.update(param_dict)
        args_real = argparse.Namespace(**args_def_dict)
        print(args_real)
        best_valids.append(main.main(args_real))
    print('Validation performances: ')
    print(best_valids)

if __name__ == '__main__':
    grid_srch()
