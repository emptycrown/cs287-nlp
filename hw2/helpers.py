import torchtext
from torchtext.vocab import Vectors
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import itertools as it


class LangEvaluator(object):
    def __init__(self, model, TEXT, **kwargs):
        self._TEXT = TEXT
        self.model = model
        self.eval_metric = kwargs.get('evalmetric', 'perplexity')
        
    def evaluate(self, test_iter, num_iter=None):
        sum_nll = 0
        cnt_nll = 0
        for i,batch in enumerate(test_iter):
            if i % 100 == 0:
                print('Iteration %d' % i)
            # Model output: [batch_size, sent_len, size_vocab]; these
            # aren't actually probabilities if the model is a Trigram,
            # but this doesn't matter.
            batch_transpose = torch.t(batch.text).contiguous() # [batch_size, sent_len]
            log_probs = self.model(batch_transpose)
            cnt_nll += batch_transpose.size()[0] * batch_transpose.size()[1]
            sum_nll += LangTrainer.loss_nll(batch_transpose.data,
                                            log_probs, mode='sum')
            if not num_iter is None and i > num_iter:
                break
            
        return np.exp(sum_nll / cnt_nll)


# THIS IS NOT THE MOST UP-TO-DATE: see HW3-models-Noah FOR THE MOST
# UP-TO-DATE VERSION OF THIS CLASS!
class LangTrainer(object):
    def __init__(self, TEXT, model, **kwargs):
        self._TEXT = TEXT
        self._model = model
        optimizer = kwargs.get('optimizer', optim.SGD)
        self._optimizer = optimizer(filter(lambda p : p.requires_grad,
                                           model.parameters()),
                                    lr=kwargs.get('lr', 0.1))
    
    # Here batch is output from a RNN/NNLM/Trigram model:
    # [..., size_vocab], and output are the real words: [...]
    @staticmethod
    def loss_nll(batch, output, mode='mean'):
        # [batch_size * sent_len, size_vocab]
        vocab_len = output.size()[-1]
        output = output.view(-1, vocab_len)
        # [batch_size * sent_len]
        batch = batch.view(-1, 1)
        batch_probs = -1 * torch.gather(output, 1, batch)
        if mode == 'mean':
            return torch.mean(batch_probs)
        else:
            return torch.sum(batch_probs)
        return
    
    @staticmethod
    def loss_perplexity(*args):
        return torch.exp(LangTrainer.loss_nll(*args))
        
