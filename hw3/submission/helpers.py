import torchtext
from torchtext.vocab import Vectors
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools as it
import time
import matplotlib.pyplot as plt


# Functions to save/load models
def save_checkpoint(mod_enc, mod_dec, filename='checkpoint.pth.tar'):
    state_dict = {'model_encoder' : mod_enc.state_dict(),
                  'model_decoder' : mod_dec.state_dict()}
    torch.save(state_dict, filename)
def load_checkpoint(filename='checkpoint.pth.tar'):
    state_dict = torch.load(filename)
    return state_dict['model_encoder'], state_dict['model_decoder']
def set_parameters(model, sv_model, cuda=True):
    for i,p in enumerate(model.parameters()):
        p.data = sv_model[list(sv_model)[i]]
    model.cuda()

# Class that NMT{Trainer/Evaluator} extends
class NMTModelUser(object):
    # Models is a list [Encoder, Decoder]
    def __init__(self, models, TEXT_SRC, TEXT_TRG, mask_src=False,
                 attention=False, reverse_enc_input=False, cuda=True):
        self._TEXT_SRC = TEXT_SRC
        self._TEXT_TRG = TEXT_TRG
        self.trg_pad = TEXT_TRG.vocab.stoi['<pad>']
        self.src_pad = TEXT_SRC.vocab.stoi['<pad>']
        print('Target padding token: %d' % self.trg_pad)
        print('Source padding token: %d' % self.src_pad)
        self.models = models
        self.mask_src = mask_src
        self.use_attention = attention
        self.record_attention = False
        self.reverse_enc_input = reverse_enc_input
        self.cuda = cuda and torch.cuda.is_available()
        if self.cuda:
            print('Using CUDA...')
        else:
            print('CUDA is unavailable...')

    def get_src_and_trg(self, batch):
        if self.reverse_enc_input:
            src_data = torch.t(batch.src.data)
            ind_rev = torch.LongTensor(np.arange(src_data.size(1) - 1, -1, -1))
            src = torch.index_select(torch.t(batch.src.data), dim=1,
                                     index=ind_rev)
            src = src.contiguous()
        else:
            src = torch.t(batch.src.data).contiguous()
        trg = torch.t(batch.trg.data)
        # Have to shift the target so we don't predict the word 
        # we see (this is ok since sentences in trg all begin 
        # with <s>)
        trg_feat = trg[:, :-1].contiguous()
        trg_lab = trg[:, 1:].contiguous()
        return (src, trg_feat, trg_lab)

    def zeros_hidden(self, batch_sz, model_num):
        num_directions = 2 if self.models[model_num].bidirectional else 1
        return torch.zeros(self.models[model_num].num_layers * num_directions, 
                           batch_sz,
                           self.models[model_num].hidden_size)

    # Ok to have self.prev_hidden apply to encoder then decoder since
    # encoder all ends before decoder starts
    def prepare_hidden(self, batch_sz, zero_out=True, model_num=0):
        if (not self.prev_hidden is None) and (not zero_out):
            pre_hidden = self.prev_hidden
        else:
            pre_hidden = (self.zeros_hidden(batch_sz, model_num) \
                          for i in range(2))
        if self.cuda:
            pre_hidden = tuple(t.cuda() for t in pre_hidden)
        return tuple(autograd.Variable(t) for t in pre_hidden)

    # kwargs can contain zero_out, model_num for prepare_hidden
    def prepare_model_inputs(self, batch, **kwargs):
        if self.cuda:
            src, trg_feat, trg_lab = \
                tuple(t.cuda() for t in self.get_src_and_trg(batch))
        else:
            src, trg_feat, trg_lab = self.get_src_and_trg(batch)

        # TODO: can comment this out (assuming it passes
        # -- just is checking batch-sz)
        assert batch.src.size(1) == batch.trg.size(1)
        var_hidden = self.prepare_hidden(batch.src.size(1), **kwargs)

        var_src = autograd.Variable(src)
        var_trg_feat = autograd.Variable(trg_feat)
        var_trg_lab = autograd.Variable(trg_lab)

        return (var_src, var_trg_feat, var_trg_lab, var_hidden)

    def init_epoch(self):
        self.prev_hidden = None
        self.debug_cnt = 0
        
    def debug_model_output(self, var_src, var_trg, dec_output,
                           num_samp=10):
        print('DEBUG CNT: %d' % self.debug_cnt)
        print(var_src.size(), var_trg.size(), dec_output.size())
        self.debug_cnt += 1
        if self.debug_cnt > 10:
            return
        src = var_src.data
        trg = var_trg.data
        _, pred = torch.topk(dec_output, k=1, dim=2)
        pred = pred.squeeze().data
        print(pred.size()) # should be [batch_sz, sent_len]
        for i in range(num_samp):
            print('=== SAMPLE %d ===' % i)
            print('-- SRC --')
            print(' '.join(self._TEXT_SRC.vocab.itos[src[i,j]] \
                                           for j in range(src.size(1))))
            print('-- REAL TRG --')
            print(' '.join(self._TEXT_TRG.vocab.itos[trg[i,j]] \
                                           for j in range(trg.size(1))))
            print('-- PRED TRG --')
            print(' '.join(self._TEXT_TRG.vocab.itos[pred[i,j]] \
                                           for j in range(pred.size(1))))


    def set_enc_prev_hidden(self, enc_hidden):
        # Each element of hidden is [F(x_T), B(x_1)]; we want to use
        # the first
        if self.models[1].enc_directions == 2:
            assert self.use_attention
            # This is the first hidden B(x_1) of the backwards layer
            self.prev_hidden = tuple(h[self.models[0].num_layers:,:,:] for \
                                     h in enc_hidden)
        else:
            self.prev_hidden = enc_hidden

    def generate_attn_mask(self, var_src):
        if not self.mask_src:
            return None
        # Using broadcasting 
        pad_mask = torch.eq(var_src, self.src_pad).type(torch.FloatTensor)
        pad_mask = pad_mask.cuda() if self.cuda else pad_mask
        return pad_mask
        
    def run_model(self, batch, mode='mean'):
        # var_src, var_trg are [batch_sz, sent_len]
        var_src, var_trg_feat, var_trg_lab, var_hidden = \
            self.prepare_model_inputs(
            batch, zero_out=True, model_num=0)

        # For attention, will use enc_output (not otherwise)
        enc_output, enc_hidden = self.models[0](var_src, var_hidden)
        self.set_enc_prev_hidden(enc_hidden)
            
        if self.use_attention:
            pad_mask = self.generate_attn_mask(var_src)
            dec_output, dec_hidden, dec_attn = self.models[1](
                var_trg_feat, self.prev_hidden, enc_output, pad_mask)
            if self.record_attention:
                _, pred = torch.topk(dec_output, k=1, dim=2)
                self.attns_log.append((dec_attn, var_src, pred.squeeze(),
                                       var_trg_lab))
        else:
            # Using real words as input. Use prev_hidden both to
            # initialize hidden state (the first time) and as context
            # vector
            dec_output, dec_hidden = self.models[1](
                var_trg_feat, self.prev_hidden, enc_hidden)
            
        # TEMPORARY
        # self.debug_model_output(var_src, var_trg_lab, dec_output)
        self.prev_hidden = dec_hidden
        loss = self.nll_loss(dec_output, var_trg_lab, mode=mode)
        return loss

    # Assume log_probs is [batch_sz, sent_len, V], output is
    # [batch_sz, sent_len]
    def nll_loss(self, log_probs, output, mode='mean', **kwargs):
        batch_sz = log_probs.size(0)
        sl_type = torch.cuda.FloatTensor if self.cuda else \
            torch.FloatTensor
        sent_len = torch.sum((output != self.trg_pad).type(sl_type)) / batch_sz
        # sent_len = sent_len.data[0]
        # sent_len = log_probs.size(1)
        # print(sent_len, log_probs.size())
        log_probs_rshp = log_probs.view(-1, log_probs.size(2))
        output_rshp = output.view(-1)
        if mode == 'mean':
            # Sum over all words in sent, mean over sentences; 
            # make sure to ignore padding
            return F.nll_loss(log_probs_rshp, output_rshp, 
                              ignore_index=self.trg_pad, 
                              **kwargs) * sent_len
        elif mode == 'sum':
            # Sum over all sentences and words in them
            return F.nll_loss(log_probs_rshp, output_rshp,
                              ignore_index=self.trg_pad,
                              size_average=False)
        else:
            raise ValueError('Invalid mode field: %s' % mode)
            
class NMTEvaluator(NMTModelUser):
    def __init__(self, models, TEXT_SRC, TEXT_TRG, record_attention=False,
                 visualize_freq=None, **kwargs):
        super(NMTEvaluator, self).__init__(models, TEXT_SRC, TEXT_TRG,
                                           **kwargs)
        # Perhaps overwrite record_attention
        self.record_attention = record_attention
        self.visualize_freq = visualize_freq
        
    def init_epoch(self):
        super(NMTEvaluator, self).init_epoch()
        self.attns_log = list()
        
    def visualize_attn(self, dec_attn_smpl, var_src_smpl, pred_smpl,
                       var_trg_lab=None, save_fn=None):
        # dec_attn_smpl is [src_len, pred_len], var_src_smpl is [src_len],
        # pred_smpl is [pred_len]
        attn = dec_attn_smpl.cpu().data.numpy()
        src_words = np.array(list(map(lambda x: self._TEXT_SRC.vocab.itos[x], 
                                      var_src_smpl.cpu().data.numpy())))
        pred_words = np.array(list(map(lambda x: self._TEXT_TRG.vocab.itos[x], 
                                       pred_smpl.cpu().data.numpy())))
        if not var_trg_lab is None:
            trg_cpu = var_trg_lab.cpu().data.numpy()
            trg_words = np.array(list(map(lambda x : self._TEXT_TRG.vocab.itos[x],
                                         trg_cpu)))
            pred_words = np.array(['%s (%s)' % (pred_words[i], trg_words[i]) for \
                                   i in range(pred_words.shape[0])])
            pad_idx = np.where(trg_words == '<pad>')[0]
            if len(pad_idx):
                clip_len = pad_idx[0]
                trg_words = trg_words[:clip_len]
                pred_words = pred_words[:clip_len]
                attn = attn[:clip_len, :]
        
        fig, ax = plt.subplots()
        ax.imshow(attn, cmap='gray')
        plt.xticks(range(len(src_words)),src_words, rotation='vertical')
        plt.yticks(range(len(pred_words)),pred_words)
        ax.xaxis.tick_top()
        if not save_fn is None:
            plt.savefig(save_fn)
        plt.show()

    def evaluate(self, test_iter, num_iter=None):
        start_time = time.time()
        for model in self.models:
            model.eval()
        nll_sum = 0
        nll_cnt = 0

        self.init_epoch()
        test_iter.init_epoch()
        for i,batch in enumerate(test_iter):
            nll_cnt += batch.trg.data.numel()
            loss = self.run_model(batch, mode='sum')
            # TODO: make sure loss just has 1 element!
            nll_sum += loss.data[0]
            
            if self.visualize_freq and i % self.visualize_freq == 0:
                sample = self.attns_log[-1]
                self.visualize_attn(sample[0][0], sample[1][0], sample[2][0])
            if not num_iter is None and i > num_iter:
                break
                        
        # Wrap the model.eval(), just in case
        for model in self.models:
            model.train()
        
        print('Validation time: %f seconds' % (time.time() - start_time))
        return np.exp(nll_sum / nll_cnt)
    
    # Performs beam search
    def run_model_predict(self, sent, ref_beam, ref_voc,
                          beam_size=100, pred_len=3, pred_num=None,
                          ignore_eos=False, translate_mode=False):
        if pred_num is None:
            pred_num = beam_size
        
        # [sent_len]
        sent_tsr = torch.LongTensor(sent)
        if self.reverse_enc_input:
            ind_rev = torch.LongTensor(np.arange(sent_tsr.size(0) - 1, -1, -1))
            sent_tsr = torch.index_select(sent_tsr, dim=0,
                                          index=ind_rev)
        if self.cuda:
            sent_tsr = sent_tsr.cuda()        
        var_src = autograd.Variable(sent_tsr.view(1, -1).expand(beam_size, -1))
        var_hidden = self.prepare_hidden(beam_size, zero_out=True)
        
        # For attention, will use enc_output (not otherwise)
        enc_output, enc_hidden = self.models[0](var_src, var_hidden)
        self.set_enc_prev_hidden(enc_hidden)
        
        # Make sure to start with SOS token
        sos_token = self._TEXT_TRG.vocab.stoi['<s>']
        self.cur_beams = (sos_token * torch.ones(beam_size, 1)).type(torch.LongTensor)
        self.cur_beam_vals = torch.zeros(beam_size, 1).type(torch.FloatTensor)
        if self.cuda:
            self.cur_beams = self.cur_beams.cuda()
            self.cur_beam_vals = self.cur_beam_vals.cuda()
        self.cur_beams = autograd.Variable(self.cur_beams)
        self.cur_beam_vals = autograd.Variable(self.cur_beam_vals)

        if translate_mode:
            final_preds = list()

        for i in range(pred_len):
            if translate_mode:
                (ref_beam, ref_voc) = self.create_ref_arrays(self.cur_beams.size(0))

            cur_sent = self.cur_beams[:, i:i+1]
            if self.use_attention:
                pad_mask = self.generate_attn_mask(var_src)

                dec_output, dec_hidden, dec_attn = self.models[1](
                    cur_sent, self.prev_hidden, enc_output[:self.cur_beams.size(0)], pad_mask)
                if self.record_attention:
                    _, pred = torch.topk(dec_output, k=1, dim=2)
                    self.attns_log.append((dec_attn, var_src, pred.squeeze(),
                                           None))
            else:
                # Using real words as input. Use prev_hidden both to
                # initialize hidden state (the first time) and as context
                # vector
                dec_output, dec_hidden = self.models[1](
                    cur_sent, self.prev_hidden, enc_hidden)
            self.prev_hidden = dec_hidden
            
            # dec_output is [batch_sz, sent_len=1, V]
            # print(dec_output.size())
            # Using broadcasting:
            dec_output = dec_output.squeeze()

            # Deal with EOS tokens:
            if ignore_eos:
                eos_token = self._TEXT_TRG.vocab.stoi['</s>']
                dec_output[:, eos_token] = -np.inf

            dec_output = dec_output + self.cur_beam_vals
            if i == 0:
                # All start words were the same, so need to restrict 
                # to the first row
                dec_output = dec_output[0, :]
            else:
                dec_output = dec_output.view(-1)
                
            topk_dec, topk_inds = torch.topk(dec_output, k=beam_size)
            chosen_prev_inds = torch.index_select(ref_beam, dim=0, index=topk_inds)
            chosen_prevs = torch.index_select(self.cur_beams, dim=0,
                                              index=chosen_prev_inds)
            # Important to update hidden to reflect which prev 
            # sents we choose
            self.prev_hidden = tuple(torch.index_select(
                    self.prev_hidden[j], dim=1, index=chosen_prev_inds) \
                                     for j in range(len(self.prev_hidden)))
            
            # Update self.cur_beam_vals: [beam_sz, 1] 
            # (we already added on prev cur_beam_vals above)
            self.cur_beam_vals = topk_dec.view(-1, 1)
            # print('cur_beam_vals', self.cur_beam_vals)

            # [batch_sz=beam_sz, 1]
            chosen_nexts = torch.index_select(ref_voc, dim=0, index=topk_inds).view(-1, 1)
            # print('chosen_nexts', chosen_nexts)
            self.cur_beams = torch.cat((chosen_prevs, chosen_nexts), dim=1)

            # Hugely inefficient, but whatever
            if translate_mode:
                keep_sents = list()
                eos_token = self._TEXT_TRG.vocab.stoi['</s>']
                for i in range(self.cur_beams.size(0)):
                    if self.cur_beams[i,-1].data[0] == eos_token:
                        final_preds.append(self.cur_beams[i,:])
                    else:
                        keep_sents.append(i)
                keep_idx = autograd.Variable(torch.LongTensor(keep_sents))
                keep_idx = keep_idx.cuda() if self.cuda else keep_idx

                # Now reselect
                if len(keep_sents) > 100 or len(keep_idx) == 0:
                    return final_preds           
                self.cur_beams = torch.index_select(self.cur_beams, 0,
                                                        keep_idx)
                self.cur_beam_vals = torch.index_select(self.cur_beam_vals, 0,
                                                            keep_idx)
                self.prev_hidden = tuple(torch.index_select(
                    self.prev_hidden[j], 1, keep_idx) \
                                             for j in range(len(self.prev_hidden)))
                # print('cur beams', self.cur_beams)
                # print('cur beam vals', self.cur_beam_vals)
                # print('prev hidden', self.prev_hidden[0].size())


            # print('cur_beams', self.cur_beams)

        if translate_mode:
            return final_preds
        return self.cur_beams
    
    @staticmethod
    def escape(l):
        return l.replace("\"", "<quote>").replace(",", "<comma>")

    def create_ref_arrays(self, beam_size):
        # Create reference idx for expanding beams and vocab
        trg_vocab_sz = len(self._TEXT_TRG.vocab)
        ref_beam = torch.LongTensor(np.arange(beam_size)).view(-1, 1).expand(-1, trg_vocab_sz)
        ref_beam = ref_beam.contiguous().view(-1)
        ref_beam = ref_beam.cuda() if self.cuda else ref_beam
        ref_beam = autograd.Variable(ref_beam)
        
        ref_voc = torch.LongTensor(np.arange(trg_vocab_sz)).view(1, -1).expand(beam_size, -1)
        ref_voc = ref_voc.contiguous().view(-1)
        ref_voc = ref_voc.cuda() if self.cuda else ref_voc
        ref_voc = autograd.Variable(ref_voc)
        return (ref_beam, ref_voc)
    
    def create_text(self, word_ids, src_or_trg, is_variable=False):
        TEXT = self._TEXT_SRC if src_or_trg == 'src' else self._TEXT_TRG
        if is_variable:
            words = [TEXT.vocab.itos[word_ids[k].data[0]] for k in range(1, len(word_ids))]
        else:
            words = [TEXT.vocab.itos[word_ids[k]] for k in range(1, len(word_ids))]
        return ' '.join(words)
        
    
    def view_predictions(self, batch_src, batch_trg, batch_preds):
        for i,sent in enumerate(batch_src):
            if i > 10:
                break
            words_src = self.create_text(sent, 'src')
            words_trg = self.create_text(batch_trg[i], 'trg')
            words_pred_list = list()
            for j in range(len(batch_preds[i])):
                words_pred_list.append(self.create_text(batch_preds[i][j], 'trg',
                                                       is_variable=True))
            print('>>> %s \n <<< %s \n ===' % (words_src, words_trg))
            for w in words_pred_list:
                print(w)
            
        
    def predict(self, test_set, fn='predictions.txt', num_cands=100, pred_len=3,
                beam_size=100, ignore_eos=False, translate_mode=False):
        start_time = time.time()
        for model in self.models:
            model.eval()

        (ref_beam, ref_voc) = self.create_ref_arrays(beam_size)
            
        self.init_epoch()
        predictions = list()
        for i,sent in enumerate(test_set):
            # [pred_num, pred_len] tensor
            best_translations = self.run_model_predict(sent, ref_beam=ref_beam,
                                                       ref_voc=ref_voc,
                                                       pred_len=pred_len,
                                                       beam_size=beam_size,
                                                       ignore_eos=ignore_eos,
                                                       translate_mode=translate_mode)
            predictions.append(best_translations)
            # if i > 10:
            #     break
            
        if translate_mode:
            return predictions
            
        print('Writing predictions to %s...' % fn)
        with open(fn, 'w') as fout:
            print('id,word', file=fout)
            for i,preds in enumerate(predictions):
                # We can traverse the beam in order since topk 
                # sorts its output
                cands = list()
                for j in range(num_cands):
                    # Ignore SOS
                    words = [self._TEXT_TRG.vocab.itos[preds[j,k].data[0]] for k in range(1, pred_len + 1)]
                    sent = '|'.join(self.escape(l) for l in words)
                    cands.append(sent)
                print('%d,%s' % (i+1, ' '.join(cands)), file=fout)
        print('Computing predictions took %f seconds' % (time.time() - start_time))
        
        # Wrap model.eavl
        for model in self.models:
            model.train()
            

    
class NMTTrainer(NMTModelUser):
    def __init__(self, models, TEXT_SRC, TEXT_TRG, lrn_rate=0.1,
                 optimizer=optim.SGD, lrn_decay='none',
                 lrn_decay_force=np.inf,
                 lrn_decay_rate=0.1,
                 clip_norm=10, **kwargs):
        super(NMTTrainer, self).__init__(models, TEXT_SRC, TEXT_TRG, **kwargs)

        self.base_lrn_rate = lrn_rate
        self.optimizer_type = optimizer
        self.init_optimizers()

        # Do learning rate decay:
        self.lr_decay_opt = lrn_decay
        self.lr_decay_force = lrn_decay_force
        if self.lr_decay_opt == 'none' or self.lr_decay_opt == 'adaptive':
            self.lambda_lr = lambda i : 1
        elif self.lr_decay_opt == 'invlin':
            decay_rate = lrn_decay_rate
            self.lambda_lr = lambda i : 1 / (1 + (i-6) * decay_rate) if i > 6 else 1
        else:
            raise ValueError('Invalid learning rate decay option: %s' \
                             % self.lr_decay_opt)
        self.schedulers = [optim.lr_scheduler.LambdaLR(optimizer,
            self.lambda_lr) for optimizer in self.optimizers]

        self.clip_norm = clip_norm
        self.init_lists()
        if self.cuda:
            for model in self.models:
                model.cuda()
                
    def init_optimizers(self):
        self.optimizers = [self.optimizer_type(filter(lambda p : p.requires_grad,
                                                      model.parameters()),
                                               lr = self.base_lrn_rate) for \
                           model in self.models]
    def init_lists(self):
        self.training_losses = list()
        self.training_norms = list()
        self.val_perfs = list()

    def get_loss_data(self, loss):
        if self.cuda:
            return loss.data.cpu().numpy()[0]
        else:
            return loss.data.numpy()[0]

    def make_recordings(self, loss, norm):
        self.training_norms.append(norm)
        self.training_losses.append(loss)

    def clip_norms(self):
        # Clip grad norm after backward but before step
        if self.clip_norm > 0:
            parameters = tuple()
            for model in self.models:
                parameters += tuple(model.parameters())
                
            # Norm clipping: returns a float
            norm = nn.utils.clip_grad_norm(
                parameters, self.clip_norm)
        else:
            norm = -1
        return norm

    def train_batch(self, batch, **kwargs):
        for model in self.models:
            model.zero_grad()
            
        loss = self.run_model(batch)
        loss.backward()

        # norms must be clipped after backward but before step
        norm = self.clip_norms()

        loss_data = self.get_loss_data(loss)
        # print('TEMP: ', loss_data, norm)
        if kwargs.get('verbose', False):
            self.make_recordings(loss_data, norm)

        for optimizer in self.optimizers:
            optimizer.step()

        # Return loss and norm (before gradient step)
        return loss_data, norm

    def init_parameters(self):
        for model in self.models:
            for p in model.parameters():
                p.data.uniform_(-0.05, 0.05)

    def train(self, torch_train_iter, le=None, val_iter=None,
              save_model_fn=None, init_parameters=True, **kwargs):
        self.init_lists()
        start_time = time.time()
        print("Innitializing parameters status: ", init_parameters)
        if init_parameters:
            self.init_parameters()

        torch_train_iter.init_epoch()
        for epoch in range(kwargs.get('num_iter', 100)):
            self.init_epoch()
            for model in self.models:
                model.train()
                
            # Learning rate decay, if any
            if self.lr_decay_opt == 'adaptive':
                if (epoch > 2 and self.val_perfs[-1] > self.val_perfs[-2]) or \
                   (epoch >= self.lr_decay_force):
                    self.base_lrn_rate = self.base_lrn_rate / 2
                    self.init_optimizers() # Looks at self.base_lrn_rate
                    print('Decaying LR to %f' % self.base_lrn_rate)
            else:
                for scheduler in self.schedulers:
                    scheduler.step()

            # TODO: LR decay
            train_iter = iter(torch_train_iter)

            for batch in train_iter:
                res_loss, res_norm = self.train_batch(batch, **kwargs)

            if epoch % kwargs.get('skip_iter', 1) == 0:
                if not kwargs.get('verbose', False):
                    self.make_recordings(res_loss, res_norm)

            print('Epoch %d, loss: %f, norm: %f, elapsed: %f, lrn_rate: %f' \
                  % (epoch, np.mean(self.training_losses[-10:]),
                     np.mean(self.training_norms[-10:]),
                     time.time() - start_time,
                     self.base_lrn_rate)) #  * self.lambda_lr(epoch)))
                    
            
            if (not le is None) and (not val_iter is None):
                self.val_perfs.append(le.evaluate(val_iter))
                print('Validation set metric: %f' % \
                      self.val_perfs[-1])

            if not save_model_fn is None:
                pathname = 'saved_models/' + save_model_fn + \
                           '.epoch_%d.ckpt.tar' % epoch
                print('Saving model to %s' % pathname)
                save_checkpoint(self.models[0], self.models[1],
                           pathname)

        if len(self.val_perfs) >= 1:
            print('FINAL VAL PERF', self.val_perfs[-1])
            return self.val_perfs[-1]
        return -1

    
