#!/bin/bash

# Scores about 184.18 after 15 epochs (kern-size=5)
# python3 main.py --tt_num_iter=100000 --early_stop --batch_sz=10 --bptt_len=32 --t_lrn_rate=0.001 --t_lrn_decay=invlin --t_lrn_decay_rate=0.1 --m_word_features=100 --t_optimizer=adam --m_kern_size_direct=5

# Scores 182.47 after 16 epochs (kern-szie=4)
# python3 main.py --tt_num_iter=100000 --early_stop --batch_sz=10 --bptt_len=32 --t_lrn_rate=0.001 --t_lrn_decay=invlin --t_lrn_decay_rate=0.1 --m_word_features=100 --t_optimizer=adam --m_kern_size_inner=4

# Scores 175.41 after 21 epochs
# python3 main.py --tt_num_iter=100000 --early_stop --batch_sz=10 --bptt_len=64 --t_lrn_rate=0.001 --t_lrn_decay=invlin --t_lrn_decay_rate=0.1 --m_word_features=100 --t_optimizer=adam --m_kern_size_inner=4

# Scores 171.00 after 34 epochs
# python3 main.py --tt_num_iter=100000 --early_stop --batch_sz=1 --bptt_len=1000 --t_lrn_rate=0.001 --t_lrn_decay=invlin --t_lrn_decay_rate=0.1 --m_word_features=100 --t_optimizer=adam --m_kern_size_inner=4
