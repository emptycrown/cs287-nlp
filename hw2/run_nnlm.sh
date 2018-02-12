#!/bin/bash

# Scores about 184.18 after 15 epochs (kern-size=5)
# python3 main.py --tt_num_iter=100000 --early_stop --batch_sz=10 --bptt_len=32 --t_lrn_rate=0.001 --t_lrn_decay=invlin --t_lrn_decay_rate=0.1 --m_word_features=100 --t_optimizer=adam --m_kern_size_direct=5

# Scores 182.47 after 16 epochs (kern-szie=4)
# python3 main.py --tt_num_iter=100000 --early_stop --batch_sz=10 --bptt_len=32 --t_lrn_rate=0.001 --t_lrn_decay=invlin --t_lrn_decay_rate=0.1 --m_word_features=100 --t_optimizer=adam --m_kern_size_inner=4

# Scores 175.41 after 21 epochs
# python3 main.py --tt_num_iter=100000 --early_stop --batch_sz=10 --bptt_len=64 --t_lrn_rate=0.001 --t_lrn_decay=invlin --t_lrn_decay_rate=0.1 --m_word_features=100 --t_optimizer=adam --m_kern_size_inner=4

# Scores 171.00 after 34 epochs (dropout = 0.25)
# python3 main.py --tt_num_iter=100000 --early_stop --batch_sz=1 --bptt_len=1000 --t_lrn_rate=0.001 --t_lrn_decay=invlin --t_lrn_decay_rate=0.1 --m_word_features=100 --t_optimizer=adam --m_kern_size_inner=4 --m_dropout=0.25

python3 main.py --network lstmlm --tt_num_iter=100000 --early_stop --bptt_len=36 --batch_sz=128 --t_lrn_rate=1.0 --t_lrn_decay=invlin --t_lrn_decay_rate=1.0 --m_word_features=100 --t_optimizer=sgd --t_clip_norm=5 --m_num_layers=2 --m_word_features=100 --m_dropout=0.5 --m_hidden_size=200


declare -a LR_BASES=(0.0005 0.001 0.002)
declare -a LR_DECAYS=(0.05 0.1 0.2)
declare -a BPTTS=(1000 2000 4000)
declare -a HIDDEN_SZS=(100 200)
declare -a WORD_FTRS=(100 200)
