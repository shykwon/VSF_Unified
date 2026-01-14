""" The code containing Trainer class and optimizer """
import torch.optim as optim
import math
from net import *
import util
import learn2learn as l2l
from Model import *


class Trainer():
    def __init__(self,args,forcast,maml,model_name, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.args = args
        self.scaler = scaler
        self.model_name = model_name
        self.forcast = forcast
        self.maml2 = maml
        self.loss = util.masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl
        self.adaptation_steps = 1
        self.model = nn.ModuleList([self.maml2, self.forcast])

        self.opt = optim.Adam(list(self.maml2.parameters())+list(self.forcast.parameters()), lr= args.learning_rate, weight_decay=args.weight_decay)

    def train(self, args, input, real_val, idx=None, id_subset = None):
        self.model.train()
        self.opt.zero_grad()
        num = input.shape[0]
        loss_forcast = torch.tensor(0.0).to(args.device)
        for i in range(num):
            clone2 = self.maml2.clone()
            task = input[i]#(1,140,12)
            batch = data_processing(task, 0.85, task.shape[1])
            task_forcast = real_val[i]
            loss1= clone2(batch)
            loss1.to(args.device)
            loss_total = loss1  #loss_DCD
            loss_total.to(args.device)
            clone2.adapt(loss_total, allow_unused=True,allow_nograd = True)

            output = torch.unsqueeze(task, dim=1)  # (1,1,140,12)-->(B,1,L,K)

            output = self.forcast(output, idx=idx, args=args)
            output = output.transpose(1, 3)  # (B,L,K,1)

            real = torch.unsqueeze(task_forcast, dim=0)
            real = torch.unsqueeze(real, dim=1)
            predict = self.scaler.inverse_transform(output)
            if self.iter % self.step == 0 and self.task_level <= self.seq_out_len:
                self.task_level += 1
            if self.cl:
                loss, _ = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
            else:
                loss, _ = self.loss(predict, real, 0.0)
            loss_forcast += loss + loss_total

        loss_forcast.backward()
        self.opt.step()
        self.iter += 1
        return loss_forcast

    def eval(self, args, input, real_val,idx = None):
        self.model.eval()
        #注意mtgnn有idx,其他没有idx
        output = self.forcast(input,args)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)

        rmse = util.masked_rmse(predict,real,0.0)[0].item()
        return loss[0].item(), rmse,

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()
        return  grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()
