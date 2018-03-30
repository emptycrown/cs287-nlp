#!/bin/bash

# VAE  --load_model_fn=vae.0.epoch_18.ckpt.tar
: 'python3 main.py --network=vaestd --batch_sz=100 --make_plots \
	   --t_lrn_rate=0.005 --t_optimizer=sgd \
	   --m_latent_dim=2 --m_hidden_dim=500 \
	   --tt_num_epochs=40 --tt_save_model_fn=vae.0 --tt_skip_epochs=4'

python3 main.py --network=gan --batch_sz=100 --make_plots \
	   --t_lrn_rate=0.1 --t_optimizer=sgd \
	   --m_latent_dim=10 --m_hidden_dim=500 \
	   --tt_num_epochs=41 --tt_skip_epochs=8 --tt_gan_k=1 --tt_save_model_fn=gan.0
# --tt_save_model_fn=gan.0

: 'python main.py --network=vaeiaf --batch_sz=100 \
	   --t_lrn_rate=0.005 --t_optimizer=sgd \
	   --m_latent_dim=10 --m_hidden_dim=200 \
	   --tt_num_epochs=20 --tt_skip_epochs=1
'
