#!/bin/bash

python3 main.py --network lstmlm --tt_num_iter=100000 --early_stop --bptt_len=36 --batch_sz=128 --t_lrn_rate=1.0 --t_lrn_decay=invlin --t_lrn_decay_rate=1.0 --m_word_features=100 --t_optimizer=sgd --t_clip_norm=5 --m_num_layers=2 --m_word_features=100 --m_dropout=0.5 --m_hidden_size=200


declare -a LR_BASES=(0.0005 0.001 0.002)
declare -a LR_DECAYS=(0.05 0.1 0.2)
declare -a BPTTS=(1000 2000 4000)
declare -a HIDDEN_SZS=(100 200)
declare -a WORD_FTRS=(100 200)
