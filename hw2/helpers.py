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

# Debugging functions

# batch is [batch_size, sent_len], is a tensor
def inspect_batch(batch, TEXT, s=''):
    print(s)
    for i in range(batch.size(0)):
        print(' '.join([TEXT.vocab.itos[j] for j in batch[i,:]]))

# Class that Lang{Trainer/Evaluator} extends
class LangModelUser(object):
    def __init__(self, model, TEXT, use_hidden=False,
                 **kwargs):
        self._TEXT = TEXT
        self.use_hidden = use_hidden
        # Amount by which to shift label in a batch (this is a bit
        # wasteful since we're therefore wasting the first
        # self.shift_label words of a batch, but oh well...); TODO:
        # this is problematic for validation/testing...
        if self.use_hidden:
            self.shift_label = 1
        else:
            self.shift_label = 0
        self.model = model
        self.cuda = kwargs.get('cuda', True) and \
            torch.cuda.is_available()
        if self.cuda:
            print('Using CUDA for evaluation...')
        else:
            print('CUDA is unavailable...')
        self.init_epoch()
            
    def init_epoch(self):
        self.prev_feature = None
        self.prev_hidden = None

    def detach_hidden(self):
        self.prev_hidden = tuple(t.data for t in self.prev_hidden)
            
    # Here batch is output from a RNN/NNLM/Trigram model:
    # [..., size_vocab], and output are the real words: [...]
    @staticmethod
    def loss_nll(batch, output, mode='mean'):
        # [batch_size * sent_len, size_vocab]
        vocab_len = output.size()[-1]
        output = output.view(-1, vocab_len)
        sent_len = batch.size(1)
        # [batch_size * sent_len]
        batch = batch.view(-1, 1)
        batch_probs = -1 * torch.gather(output, 1, 
                                        batch) #.type(torch.LongTensor))
        if mode == 'mean':
            return torch.mean(batch_probs) * sent_len
        else:
            return torch.sum(batch_probs)
        return
    
    @staticmethod
    def loss_perplexity(*args):
        return torch.exp(self.loss_nll(*args))

    def get_feature_and_label(self, batch):
        batch_transpose = torch.t(batch.text.data)
        if self.shift_label > 0:
            if self.prev_feature is None:
                feat = batch_transpose[:,:-self.shift_label]
                lab = batch_transpose[:, self.shift_label:]
            else:
                feat = torch.cat((self.prev_feature,
                                  batch_transpose[:,:-self.shift_label]),
                                 dim=1)
                lab = batch_transpose
            self.prev_feature = batch_transpose[:,-self.shift_label:]
            return (feat.contiguous(), lab.contiguous())
        else:
            return (batch_transpose.contiguous(),
                    batch_transpose.contiguous())

    '''
    # Ignore the last self.shift_label words in each sentence
    def get_feature(self, batch):
        batch_transpose = torch.t(batch.text.data)
        if self.shift_label > 0:
            self.prev_feature = batch_transpose[:,-self.shift_label:]
            return torch.cat(
                (self.prev_feature,
                 batch_transpose[:, :-self.shift_label])).contiguous()
        else:
            return batch_transpose.contiguous()

    # Ignore the first self.shift_label words in each sentence
    def get_label(self, batch):
        batch_transpose = torch.t(batch.text.data)
        return batch_transpose[:, self.shift_label:].contiguous()
    '''

    def zeros_hidden(self, batch_sz):
        return torch.zeros(self.model.num_layers, batch_sz,
                           self.model.hidden_dim)
    
    # We haven't yet transposed batch, so this should work (and
    # batch_first does not apply to hidden layers in lstm, for some
    # reason)
    def prepare_hidden(self, batch_sz, from_scratch=False):
        if (not self.prev_hidden is None) and not from_scratch:
            pre_hidden= self.prev_hidden
        else:
            pre_hidden = (self.zeros_hidden(batch_sz), self.zeros_hidden(batch_sz))
        if self.cuda:
            pre_hidden = tuple(t.cuda() for t in pre_hidden)
        return tuple(autograd.Variable(t) for t in pre_hidden)
            
    def prepare_model_inputs(self, batch):
        # TODO: this might break trigram stuff (easy to fix)...
        if self.cuda:
            # [batch_size, sent_len]                
            feature, label = tuple(t.cuda() for t in self.get_feature_and_label(batch))
        else:
            feature, label = self.get_feature_and_label(batch)

        if self.use_hidden:
            # [num_layers, batch-sz, hidden_dim]
            var_hidden = self.prepare_hidden(batch.text.size(1))

        # print('FEATURE BATCH')
        # inspect_batch(feature, self._TEXT)
        # print('LABEL BATCH')
        # inspect_batch(label, self._TEXT)
        
        var_feature = autograd.Variable(feature)
        var_label = autograd.Variable(label)
        if self.use_hidden:
            return ([var_feature, var_hidden], var_label)
        else:
            return ([var_feature], var_label)

    # returns list of lists, each containing top 20 predictions
    def process_model_output(self, log_probs):
        # log_probs is [batch_sz, sent_len, vocab_sz]        
        target_probs = log_probs[:,-1,:]
        # pred_idx is [batch_sz, 20]
        _, pred_idx = torch.topk(target_probs, 25, dim=1)
        
        batch_pred = []
        for sent_pred in pred_idx.data:
            filtered_pred = [self._TEXT.vocab.itos[i] for i in sent_pred if self._TEXT.vocab.itos[i] != "<eos>"]
            batch_pred.append(filtered_pred[:20])
        return batch_pred

        
            

# use_hidden is special: only intended for RNN models!
class LangEvaluator(LangModelUser):
    def __init__(self, model, TEXT, use_hidden=False,
                 **kwargs):
        super(LangEvaluator, self).__init__(model, TEXT, use_hidden=use_hidden,
                                            **kwargs)
        self.eval_metric = kwargs.get('evalmetric', 'perplexity')
        
    def evaluate(self, test_iter, num_iter=None):
        start_time = time.time()
        self.model.eval() # In case we have dropout
        sum_nll = 0
        cnt_nll = 0

        for i,batch in enumerate(test_iter):
            # Model output: [batch_size, sent_len, size_vocab]; these
            # aren't actually probabilities if the model is a Trigram,
            # but this doesn't matter.
            # If self.prev_hidden is None, hidden will be 0
            var_feature_arr, var_label = self.prepare_model_inputs(batch)
            
            if self.use_hidden:
                log_probs, self.prev_hidden = self.model(*var_feature_arr)
                self.detach_hidden()
            else:
                log_probs = self.model(*var_feature_arr)
                
            # This is the true feature (might have hidden at 1)            
            cnt_nll += var_label.data.size()[0] * \
                       var_label.data.size()[1]
            sum_nll += self.loss_nll(var_label,
                                     log_probs, mode='sum').data[0]
            if not num_iter is None and i > num_iter:
                break

        self.model.train() # Wrap model.eval()
        print('Validation time: %f seconds' % (time.time() - start_time))
        return np.exp(sum_nll / cnt_nll)
    
    def predict(self, test_set):
        start_time = time.time()
        self.model.eval() # In case we have dropout

        predictions = []
        for i,sent in enumerate(test_set):
            # Model output: [1, sent_len, size_vocab]; 
            var_feature = autograd.Variable(torch.LongTensor(sent).cuda()).view(1,-1)
            var_hidden = self.prepare_hidden(1, from_scratch=True)
            var_feature_arr = [var_feature, var_hidden]
            
            if self.use_hidden:
                # keeping track of hidden state does not matter in test set
                log_probs, _ = self.model(*var_feature_arr)
            else:
                log_probs = self.model(*var_feature_arr)

            # Get predictions for test data
            predictions += self.process_model_output(log_probs)
                
        print('Writing test predictions to predictions.txt...')
        with open("predictions_eos.txt", "w") as fout: 
            print("id,word", file=fout)
            for i,l in enumerate(predictions, 1):
                print("%d,%s"%(i, " ".join(l)), file=fout)

        self.model.train() # Wrap model.eval()
        print('Validation time: %f seconds' % (time.time() - start_time))
        return predictions

# Option use_hidden is special: only intended for LSTM/RNN usage!
class LangTrainer(LangModelUser):
    def __init__(self, TEXT, model, use_hidden=False,
                 **kwargs):
        super(LangTrainer, self).__init__(model, TEXT, use_hidden=use_hidden,
                                          **kwargs)
        # Settings:
        self.base_lrn_rate = kwargs.get('lrn_rate', 0.1)
        self.optimizer_type = kwargs.get('optimizer', optim.SGD)
        self._optimizer = self.optimizer_type(filter(lambda p : p.requires_grad,
                                                     model.parameters()),
                                              lr=self.base_lrn_rate)

        # Do learning rate decay:
        self.lr_decay_opt = kwargs.get('lrn_decay', 'none')
        if self.lr_decay_opt == 'none' or self.lr_decay_opt == 'adaptive':
            self.lambda_lr = lambda i : 1
        elif self.lr_decay_opt == 'invlin':
            decay_rate = kwargs.get('lrn_decay_rate', 0.1)
            self.lambda_lr = lambda i : 1 / (1 + (i-6) * decay_rate) if i > 6 else 1
        else:
            raise ValueError('Invalid learning rate decay option: %s' \
                             % self.lr_decay_opt)
        self.scheduler = optim.lr_scheduler.LambdaLR(self._optimizer,
                                                     self.lambda_lr)
            
        self.clip_norm = kwargs.get('clip_norm', 5)
        self.init_lists()
        if self.cuda:
            self.model.cuda()
    
    def init_lists(self):
        self.training_losses = list()
        self.training_norms = list()
        self.val_perfs = list()
            
    # We are doing a slightly funky thing of taking a 
    # variable's data and then making a new 
    # variable...this is kinda unnecessary and ugly
    def make_loss(self, batch):
        # If self.prev_hidden does not exist, hidden will be 0
        var_feature_arr, var_label = self.prepare_model_inputs(batch)
        # inspect_batch(var_feature_arr[0].data, self._TEXT, 'FEATURE')
        # inspect_batch(var_label.data, self._TEXT, 'LABEL')
        if self.use_hidden:
            log_probs, self.prev_hidden = self.model(*var_feature_arr)
            self.detach_hidden()
        else:
            log_probs = self.model(*var_feature_arr)
        loss = self.loss_nll(var_label, log_probs)
        return loss

    def make_recordings(self, loss, norm):
        if self.cuda:
            self.training_losses.append(loss.data.cpu().numpy()[0])
        else:
            self.training_losses.append(loss.data.numpy()[0])
        self.training_norms.append(norm)
    
    # le is a LangEvaluator, and if supplied must have a val_iter
    # along with it
    def train(self, torch_train_iter, le=None, val_iter=None, test_set=None,
              **kwargs):
        self.init_lists()
        start_time = time.time()
        retain_graph = kwargs.get('retain_graph', False)
        for p in self.model.parameters():
            p.data.uniform_(-0.05, 0.05)
        for epoch in range(kwargs.get('num_iter', 100)):
            self.init_epoch()
            self.model.train()
            # Learning rate decay, if any
            if self.lr_decay_opt == 'adaptive':
                if epoch > 15 and self.val_perfs[-1] > self.val_perfs[-2] - 1:
                    self.base_lrn_rate = self.base_lrn_rate / 2
                    self._optimizer = self.optimizer_type(filter(lambda p : p.requires_grad,
                                                                 self.model.parameters()),
                                                          lr=self.base_lrn_rate)
                    print('Decaying LR to %f' % self.base_lrn_rate)
            else:
                self.scheduler.step()
                
            torch_train_iter.init_epoch()
            train_iter = iter(torch_train_iter)

            for batch in train_iter:
                self.model.zero_grad()
                loss = self.make_loss(batch)
                    
                # Do gradient updates
                loss.backward(retain_graph=retain_graph)

                # Clip grad norm after backward but before step
                if self.clip_norm > 0:
                    # Norm clipping: returns a float
                    norm = nn.utils.clip_grad_norm(
                        self.model.parameters(), self.clip_norm)
                    self.training_norms.append(norm)
                else:
                    norm = -1
                    self.training_norms.append(-1)

                if kwargs.get('verbose', False):
                    self.make_recordings(loss, norm)
                self._optimizer.step()

            # Logging, early stopping
            if epoch % kwargs.get('skip_iter', 1) == 0:
                # Update training_losses
                if not kwargs.get('verbose', False):
                    self.make_recordings(loss, norm)
                # Logging
                print('Epoch %d, loss: %f, norm: %f, elapsed: %f, lrn_rate: %f' \
                      % (epoch, np.mean(self.training_losses[-10:]),
                         self.training_norms[-1],
                         time.time() - start_time,
                         self.base_lrn_rate * self.lambda_lr(epoch)))
                if (not le is None) and (not val_iter is None):
                    self.val_perfs.append(le.evaluate(val_iter))
                    print('Validation set metric: %f' % \
                          self.val_perfs[-1])
                    # We've stopped improving (basically), so stop training
                    # if len(self.val_perfs) > 2 and \
                    #    self.val_perfs[-1] > self.val_perfs[-2] - 0.5: #TODO: Change back to 0.1
                    #     break

        if kwargs.get('produce_predictions',False):
            if (not le is None) and (not test_set is None):
                print('Predicting test set...')
                print('Produced %d predictions!' % len(le.predict(test_set)))

        if len(self.val_perfs) > 1:
            print('FINAL VALID PERF', self.val_perfs[-1])
            return self.val_perfs[-1]
        return 0
