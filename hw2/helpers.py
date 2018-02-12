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
def inspect_batch(batch, TEXT):
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
            
    # Here batch is output from a RNN/NNLM/Trigram model:
    # [..., size_vocab], and output are the real words: [...]
    @staticmethod
    def loss_nll(batch, output, mode='mean'):
        # [batch_size * sent_len, size_vocab]
        vocab_len = output.size()[-1]
        output = output.view(-1, vocab_len)
        # [batch_size * sent_len]
        batch = batch.view(-1, 1)
        batch_probs = -1 * torch.gather(output, 1, 
                                        batch) #.type(torch.LongTensor))
        if mode == 'mean':
            return torch.mean(batch_probs)
        else:
            return torch.sum(batch_probs)
        return
    
    @staticmethod
    def loss_perplexity(*args):
        return torch.exp(self.loss_nll(*args))

    # Ignore the last self.shift_label words in each sentence
    def get_feature(self, batch):
        batch_transpose = torch.t(batch.text.data)
        if self.shift_label > 0:
            return batch_transpose[:, :-self.shift_label].contiguous()
        else:
            return batch_transpose.contiguous()

    # Ignore the first self.shift_label words in each sentence
    def get_label(self, batch):
        batch_transpose = torch.t(batch.text.data)
        return batch_transpose[:, self.shift_label:].contiguous()

    # We haven't yet transposed batch, so this should work (and
    # batch_first does not apply to hidden layers in lstm, for some
    # reason)
    def prepare_hidden(self, batch):
        return torch.zeros(self.model.num_layers, batch.text.size(1),
                           self.model.hidden_dim)

    def prepare_model_inputs(self, batch):
        # TODO: this might break trigram stuff (easy to fix)...
        if self.cuda:
            # [batch_size, sent_len]                
            feature, label = self.get_feature(batch).cuda(), \
                             self.get_label(batch).cuda()
            if self.use_hidden:
                # [batch_sz, num_layers, hidden_dim]
                hidden = (self.prepare_hidden(batch).cuda(),
                          self.prepare_hidden(batch).cuda())
        else:
            feature, label = self.get_feature(batch), \
                             self.get_label(batch)
            if self.use_hidden:
                hidden = (self.prepare_hidden(batch),
                          self.prepare_hidden(batch))

        # print('FEATURE BATCH')
        # inspect_batch(feature, self._TEXT)
        # print('LABEL BATCH')
        # inspect_batch(label, self._TEXT)
        
        var_feature = autograd.Variable(feature)
        var_label = autograd.Variable(label)
        if self.use_hidden:
            var_hidden = (autograd.Variable(hidden[0]),
                          autograd.Variable(hidden[1]))
            return ([var_feature, var_hidden], var_label)
        else:
            return ([var_feature], var_label)

    def process_model_output(self, log_probs):
        print(log_probs.data)
        print(log_probs.data.size())
        return "<>"

        
            

# use_hidden is special: only intended for RNN models!
class LangEvaluator(LangModelUser):
    def __init__(self, model, TEXT, use_hidden=False,
                 **kwargs):
        super(LangEvaluator, self).__init__(model, TEXT, use_hidden=use_hidden,
                                            **kwargs)
        self.eval_metric = kwargs.get('evalmetric', 'perplexity')
        
    def evaluate(self, test_iter, num_iter=None, produce_predictions=False):
        start_time = time.time()
        self.model.eval() # In case we have dropout
        sum_nll = 0
        cnt_nll = 0

        predictions = []
        for i,batch in enumerate(test_iter):
            # Model output: [batch_size, sent_len, size_vocab]; these
            # aren't actually probabilities if the model is a Trigram,
            # but this doesn't matter.
            
            var_feature_arr, var_label = self.prepare_model_inputs(batch)
            
            if self.use_hidden:
                log_probs, _ = self.model(*var_feature_arr)
            else:
                log_probs = self.model(*var_feature_arr)

            predictions.append(self.process_model_output(log_probs))
                
            # This is the true feature (might have hidden at 1)            
            var_feature = var_feature_arr[0] 
            cnt_nll += var_feature.data.size()[0] * \
                       var_feature.data.size()[1]
            sum_nll += self.loss_nll(var_feature,
                                     log_probs, mode='sum').data[0]
            if not num_iter is None and i > num_iter:
                break

        if produce_predictions:
            with open("predictions.txt", "w") as fout: 
                print("id,word", file=fout)
                for i, l in enumerate(predictions, 1):
                    print("%d,%s"%(i, " ".join(l)), file=fout)

        self.model.train() # Wrap model.eval()
        print('Validation time: %f seconds' % (time.time() - start_time))
        return np.exp(sum_nll / cnt_nll)

# Option use_hidden is special: only intended for LSTM/RNN usage!
class LangTrainer(LangModelUser):
    def __init__(self, TEXT, model, use_hidden=False,
                 **kwargs):
        super(LangTrainer, self).__init__(model, TEXT, use_hidden=use_hidden,
                                          **kwargs)
        # Settings:
        self.base_lrn_rate = kwargs.get('lrn_rate', 0.1)
        optimizer = kwargs.get('optimizer', optim.SGD)
        self._optimizer = optimizer(filter(lambda p : p.requires_grad,
                                           model.parameters()),
                                    lr=self.base_lrn_rate)

        # Do learning rate decay:
        lr_decay_opt = kwargs.get('lrn_decay', 'none')
        if lr_decay_opt == 'none':
            self.lambda_lr = lambda i : 1
        elif lr_decay_opt == 'invlin':
            decay_rate = kwargs.get('lrn_decay_rate', 0.1)
            self.lambda_lr = lambda i : 1 / (1 + i * decay_rate)
        else:
            raise ValueError('Invalid learning rate decay option: %s' \
                             % lr_decay_opt)
        self.scheduler = optim.lr_scheduler.LambdaLR(self._optimizer,
                                                     self.lambda_lr)
            
        self.clip_norm = kwargs.get('clip_norm', 5)
            
        # TODO: implement validation thing for early stopping
        self.training_losses = list()
        self.training_norms = list()
        self.val_perfs = list()
        if self.cuda:
            self.model.cuda()
    
    
    # We are doing a slightly funky thing of taking a 
    # variable's data and then making a new 
    # variable...this is kinda unnecessary and ugly
    def make_loss(self, batch):
        var_feature_arr, var_label = self.prepare_model_inputs(batch)
        if self.use_hidden:
            log_probs, _ = self.model(*var_feature_arr)
        else:
            log_probs = self.model(*var_feature_arr)
        loss = self.loss_nll(var_label, log_probs)
        return loss

    # le is a LangEvaluator, and if supplied must have a val_iter
    # along with it
    def train(self, torch_train_iter, le=None, val_iter=None, test_iter=None,
              **kwargs):
        start_time = time.time()
        retain_graph = kwargs.get('retain_graph', False)
        for epoch in range(kwargs.get('num_iter', 100)):
            self.model.train()
            # Learning rate decay, if any
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
                    self.training_norms.append(-1)

                self._optimizer.step()

            # Logging, early stopping
            if epoch % kwargs.get('skip_iter', 1) == 0:
                # Update training_losses
                if self.cuda:
                    self.training_losses.append(loss.data.cpu().numpy()[0])
                else:
                    self.training_losses.append(loss.data.numpy()[0])

                # Logging
                print('Epoch %d, loss: %f, norm: %f, elapsed: %f, lrn_rate: %f' \
                      % (epoch, self.training_losses[-1],
                         self.training_norms[-1],
                         time.time() - start_time,
                         self.base_lrn_rate * self.lambda_lr(epoch)))
                if (not le is None) and (not val_iter is None):
                    self.val_perfs.append(le.evaluate(val_iter))
                    print('Validation set metric: %f' % \
                          self.val_perfs[-1])
                    # We've stopped improving (basically), so stop training
                    if len(self.val_perfs) > 2 and \
                       self.val_perfs[-1] > self.val_perfs[-2] - 100: #TODO: Change back to 0.1
                        break
        if kwargs.get('produce_predictions',False):
            if (not le is None) and (not test_iter is None):
                print('Writing test predictions to predictions.txt...')
                print('Test set metric: %f' % \
                    le.evaluate(test_iter, produce_predictions=kwargs.get('produce_predictions',False)))

        if len(self.val_perfs) > 1:
            print('FINAL VALID PERF', self.val_perfs[-1])
            return self.val_perfs[-1]
        return 0
