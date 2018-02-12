#!/bin/bash

python3 main.py --network=lstmlm --t_retain_graph --tt_num_iter=100000 --early_stop --batch_sz=10 --bptt_len=32 --t_lrn_rate=0.001 --t_lrn_decay=invlin --t_lrn_decay_rate=0.1 --m_word_features=100 --t_optimizer=adam --m_kern_size_direct=5

declare -a LR_BASES=(0.0005 0.001 0.002)
declare -a LR_DECAYS=(0.05 0.1 0.2)
declare -a BPTTS=(1000 2000 4000)
declare -a HIDDEN_SZS=(100 200)
declare -a WORD_FTRS=(100 200)