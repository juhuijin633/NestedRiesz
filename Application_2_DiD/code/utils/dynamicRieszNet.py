import os
import copy
import numpy as np
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from pathlib import Path
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r
        
net_a_settings_global = {
    'test_split' : 0,
    'learner_lr' : 1e-4,
    'learner_l2' : 1e-3,
    'learner_l1' : 0,
    'n_epochs' : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta' : 1e-3,
    'bs' : 64,
    'optimizer' : 'adam',
    'warm_start' : False,
    'logger' : None,
    'model_dir' : '.',
    'device' : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden' : 100,
    'drop_prob' : 0,
    'degree' : 2,
    'interaction_only' : True,
    'n_common' : 200,
    'act_func' : 'elu'
}
net_f_settings_global = {
    'test_split' : 0,
    'learner_lr' : 1e-4,
    'learner_l2' : 1e-3,
    'learner_l1' : 0,
    'n_epochs' : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta' : 1e-3,
    'bs' : 64,
    'optimizer' : 'adam',
    'warm_start' : False,
    'logger' : None,
    'model_dir' : '.',
    'device' : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden' : 100,
    'drop_prob' : 0,
    'degree' : 2,
    'interaction_only' : True,
    'n_common' : 200,
    'act_func' : 'elu'
}

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

def _combinations(n_features, degree, interaction_only):
        comb = (combinations if interaction_only else combinations_w_r)
        return chain.from_iterable(comb(range(n_features), i)
                                   for i in range(0, degree + 1))

def L1_reg(net, l1_value, skip_list=()):
    L1_reg_loss = 0.0
    for name, param in net.named_parameters():
        if not param.requires_grad or len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            continue  # frozen weights
        else:
            L1_reg_loss += torch.sum(abs(param))
    L1_reg_loss *= l1_value
    return L1_reg_loss

class Learner_a_Net:
    def __init__(self, net_a_settings = net_a_settings_global):
        """
        """
        self.net_a_settings = net_a_settings
        self.test_split = net_a_settings['test_split']
        self.n_hidden = net_a_settings['n_hidden']
        self.drop_prob = net_a_settings['drop_prob']

    def fit(self, X, D, a_prev):

        d = D * 0
        
        DX = torch.hstack((D, X))
        dX = torch.hstack((d, X))

        if self.test_split > 0:
            DX_train, DX_test, dX_train, dX_test, a_prev_train, a_prev_test = train_test_split(DX, dX, a_prev, test_size = self.test_split)
        else:
            DX_train, dX_train, a_prev_train = DX, dX, a_prev
            DX_test, dX_test, a_prev_test = None, None, None

        if self.net_a_settings['act_func'] == 'leaky':
            self.learner = RieszLearner2(DX.shape[1], self.n_hidden, self.drop_prob, 0, interaction_only=True)
        else:
            self.learner = RieszLearner(DX.shape[1], self.n_hidden, self.drop_prob, 0, interaction_only=True)

        self.learner_a = RieszNet(self.learner)
        
        # Settings
        train_opt = {'earlystop_rounds' : self.net_a_settings['earlystop_rounds'],
                         'earlystop_delta' : self.net_a_settings['earlystop_delta'],
                            'learner_lr' : self.net_a_settings['learner_lr'],
                                'learner_l2' : self.net_a_settings['learner_l2'],
                                    'learner_l1' : self.net_a_settings['learner_l1'],
                                        'n_epochs' : self.net_a_settings['n_epochs'],
                                            'bs' : self.net_a_settings['bs'],
                                                'optimizer' : self.net_a_settings['optimizer']} 

        # training
        self.learner_a.fit(DX_train, dX_train, a_prev_train, DXval=DX_test, dXval=dX_test, a_prevval = a_prev_test,
                **train_opt,
                model_dir=str(Path.home()), device=self.net_a_settings['device'])

    def predict(self, X, D, *, model='final'):
        DX = torch.hstack((D, X))
        output = self.learner_a.predict(DX, model=model)
        if not torch.is_tensor(output):
            output = torch.tensor(output).float()
        return output

class RieszLearner(nn.Module):

    def __init__(self, n_t, n_hidden, p, degree, interaction_only=False):
        super().__init__()
        n_common = 200
        self.monomials = list(_combinations(n_t, degree, interaction_only))
        self.common = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU())
        self.riesz_nn = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_poly = nn.Sequential(nn.Linear(len(self.monomials), 1))


    def forward(self, x):
        poly = torch.cat([torch.prod(x[:, t], dim=1, keepdim=True)
                          for t in self.monomials], dim=1)
        feats = self.common(x)
        riesz = self.riesz_nn(feats) + self.riesz_poly(poly)
        
        return riesz

class RieszLearner2(nn.Module):

    def __init__(self, n_t, n_hidden, p, degree, interaction_only=False):
        super().__init__()
        n_common = 200
        self.monomials = list(_combinations(n_t, degree, interaction_only))
        self.common = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_common), nn.LeakyReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.LeakyReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.LeakyReLU())
        self.riesz_nn = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_poly = nn.Sequential(nn.Linear(len(self.monomials), 1))

    def forward(self, x):
        poly = torch.cat([torch.prod(x[:, t], dim=1, keepdim=True)
                          for t in self.monomials], dim=1)
        feats = self.common(x)
        riesz = self.riesz_nn(feats) + self.riesz_poly(poly)

        return riesz

class RieszNet:

    def __init__(self, learner):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        """
        self.learner = learner

    def _pretrain(self, DX, dX, a_prev, DXval, dXval, a_prevval, *, bs,
                  warm_start, logger, model_dir, device):
        """ Prepares the variables required to begin training.
        """
        

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name
        self.device = device

        if not torch.is_tensor(DX):
            DX = torch.Tensor(DX).to(self.device)
        if not torch.is_tensor(dX):
            dX = torch.Tensor(dX).to(self.device)
        if not torch.is_tensor(a_prev):
            a_prev = torch.Tensor(a_prev).to(self.device)
        if (DXval is not None) and (not torch.is_tensor(DXval)):
            DXval = torch.Tensor(DXval).to(self.device)
        if (dXval is not None) and (not torch.is_tensor(dXval)):
            dXval = torch.Tensor(dXval).to(self.device)
        if (a_prevval is not None) and (not torch.is_tensor(a_prevval)):
            a_prevval = torch.Tensor(a_prevval).to(self.device)

        indices = torch.arange(DX.size(0)).to(self.device)
        self.train_ds = TensorDataset(DX, indices)
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        self.learner = self.learner.to(device)

        if not warm_start:
            self.learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        self.logger = logger
        if self.logger is not None:
            self.writer = SummaryWriter()

        return DX, dX, a_prev, DXval, dXval, a_prevval

    def _train(self, DX, dX, a_prev, *, DXval=None, dXval = None, a_prevval=None,
               earlystop_rounds, earlystop_delta,
               learner_l2, learner_l1, learner_lr,
               n_epochs, bs,
               optimizer):

        parameters = add_weight_decay(self.learner, learner_l2)
        if optimizer == 'adam':
            self.optimizerD = optim.Adam(parameters, lr=learner_lr)
        elif optimizer == 'rmsprop':
            self.optimizerD = optim.RMSprop(parameters, lr=learner_lr, momentum=.9)
        elif optimizer == 'sgd':
            self.optimizerD = optim.SGD(parameters, lr=learner_lr, momentum=.9, nesterov=True)
        else:
            raise AttributeError("Not implemented")

        if DXval is not None:
            min_eval = np.inf
            time_since_last_improvement = 0
            best_learner_state_dict = copy.deepcopy(self.learner.state_dict())
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, mode='min', factor=0.5,
                patience=5, threshold=0.0, threshold_mode='abs', cooldown=0, min_lr=0,
                eps=1e-08)

        for epoch in range(n_epochs):


            for it, (dx, ind) in enumerate(self.train_dl):

                self.learner.train()
                output = self.learner(dx)

                L1_reg_loss = 0.0
                if learner_l1 > 0.0:
                    L1_reg_loss = L1_reg(self.learner, learner_l1)

                D_loss = torch.mean(- 2 * a_prev[ind,:] * self.learner(dX[ind,:]) + output ** 2) + L1_reg_loss

                self.optimizerD.zero_grad()
                D_loss.backward()
                self.optimizerD.step()
                self.learner.eval()

            if DXval is not None:  # if early stopping was enabled we check the out of sample violation
                output = self.learner(DXval)

                self.curr_eval = torch.mean(- 2 * a_prevval * self.learner(dXval) + output ** 2)

                lr_scheduler.step(self.curr_eval)

                if min_eval > self.curr_eval + earlystop_delta:
                    min_eval = self.curr_eval
                    time_since_last_improvement = 0
                    best_learner_state_dict = copy.deepcopy(
                        self.learner.state_dict())
                else:
                    time_since_last_improvement += 1
                    if time_since_last_improvement > earlystop_rounds:
                        break

            if self.logger is not None:
                self.logger(self, self.learner, epoch, self.writer)

        torch.save(self.learner, os.path.join(
            self.model_dir, "epoch{}".format(epoch)))

        self.n_epochs = epoch + 1
        if DXval is not None:
            self.learner.load_state_dict(best_learner_state_dict)
            torch.save(self.learner, os.path.join(
                self.model_dir, "earlystop"))

        return self

    def fit(self, DX, dX, a_prev, DXval=None, dXval = None, a_prevval = None, *,
            earlystop_rounds=20, earlystop_delta=0,
            learner_l2=1e-3, learner_l1=0, learner_lr=0.001,
            n_epochs=100, bs=100, optimizer='adam',
            warm_start=False, logger=None, model_dir='.', device=None):

        DX, dX, a_prev, DXval, dXval, a_prevval = self._pretrain(DX, dX, a_prev, DXval, dXval, a_prevval, bs=bs, warm_start=warm_start,
                                 logger=logger, model_dir=model_dir,
                                 device=device)

        self._train(DX, dX, a_prev, DXval=DXval, dXval=dXval, a_prevval=a_prevval,
                    earlystop_rounds=earlystop_rounds, earlystop_delta=earlystop_delta,
                    learner_l2=learner_l2, learner_l1=learner_l1,
                    learner_lr=learner_lr, n_epochs=n_epochs, bs=bs,
                    optimizer=optimizer)

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self

    def get_model(self, model):
        if model == 'final':
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(self.n_epochs - 1)),
                                           weights_only = False)
        if model == 'earlystop':
            return torch.load(os.path.join(self.model_dir,
                                           "earlystop"),  weights_only = False)

        raise AttributeError("Not implemented")

    def predict(self, X, model='final'):

        return self.get_model(model)(X).cpu().data

class Learner_f_Net:
    def __init__(self, net_f_settings = net_f_settings_global):
        """
        """
        self.net_f_settings = net_f_settings
        self.test_split = net_f_settings['test_split']
        self.n_hidden = net_f_settings['n_hidden']
        self.drop_prob = net_f_settings['drop_prob']

    def fit(self, y, X, D):

        DX = torch.hstack((D, X))

        if self.test_split > 0:
            DX_train, DX_test, y_train, y_test = train_test_split(DX, y, test_size = self.test_split)
        else:
            DX_train, y_train = DX, y
            DX_test, y_test = None, None

        self.learner = RegLearner(DX.shape[1], self.n_hidden, self.drop_prob, 0, interaction_only=True)
        self.learner_f = RegNet(self.learner)
        
        # Settings
        train_opt = {'earlystop_rounds' : self.net_f_settings['earlystop_rounds'],
                         'earlystop_delta' : self.net_f_settings['earlystop_delta'],
                            'learner_lr' : self.net_f_settings['learner_lr'],
                                'learner_l2' : self.net_f_settings['learner_l2'],
                                    'learner_l1' : self.net_f_settings['learner_l1'],
                                        'n_epochs' : self.net_f_settings['n_epochs'],
                                            'bs' : self.net_f_settings['bs'],
                                                'optimizer' : self.net_f_settings['optimizer']} 

        # training
        self.learner_f.fit(DX_train, y_train, Xval= DX_test, yval = y_test,
                **train_opt,
                model_dir=str(Path.home()), device=self.net_f_settings['device'])

    def predict(self, X, D, *, model='final'):
        DX = torch.hstack((D, X))
        output = self.learner_f.predict(DX, model=model)
        if not torch.is_tensor(output):
            output = torch.tensor(output).float()
        return output

class RegLearner(nn.Module):

    def __init__(self, n_t, n_hidden, p, degree, interaction_only=False):
        super().__init__()
        n_common = 200
        self.monomials = list(_combinations(n_t, degree, interaction_only))
        self.common = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU())
        self.reg_nn0 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        self.reg_nn1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        self.reg_poly = nn.Sequential(nn.Linear(len(self.monomials), 1))

    def forward(self, x):
        poly = torch.cat([torch.prod(x[:, t], dim=1, keepdim=True)
                          for t in self.monomials], dim=1)
        feats = self.common(x)
        reg = self.reg_nn0(feats) * (1 - x[:, [0]]) + self.reg_nn1(feats) * x[:, [0]] + self.reg_poly(poly)
        return reg

class RegNet:

    def __init__(self, learner):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        """
        self.learner = learner

    def _pretrain(self, X, y, Xval, yval, *, bs,
                  warm_start, logger, model_dir, device):
        """ Prepares the variables required to begin training.
        """
        

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name
        self.device = device

        if not torch.is_tensor(X):
            X = torch.Tensor(X).to(self.device)
        if not torch.is_tensor(y):
            y = torch.Tensor(y).to(self.device)
        if (Xval is not None) and (not torch.is_tensor(Xval)):
            Xval = torch.Tensor(Xval).to(self.device)
        if (yval is not None) and (not torch.is_tensor(yval)):
            yval = torch.Tensor(yval).to(self.device)
            yval = yval.reshape(-1, 1)

        y = y.reshape(-1, 1)

        self.train_ds = TensorDataset(X, y)
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        self.learner = self.learner.to(device)

        if not warm_start:
            self.learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        self.logger = logger
        if self.logger is not None:
            self.writer = SummaryWriter()

        return X, y, Xval, yval

    def _train(self, X, y, *, Xval, yval,
               earlystop_rounds, earlystop_delta,
               learner_l2, learner_l1, learner_lr,
               n_epochs, bs, 
               optimizer):

        parameters = add_weight_decay(self.learner, learner_l2)
        if optimizer == 'adam':
            self.optimizerD = optim.Adam(parameters, lr=learner_lr)
        elif optimizer == 'rmsprop':
            self.optimizerD = optim.RMSprop(parameters, lr=learner_lr, momentum=.9)
        elif optimizer == 'sgd':
            self.optimizerD = optim.SGD(parameters, lr=learner_lr, momentum=.9, nesterov=True)
        else:
            raise AttributeError("Not implemented")

        if Xval is not None:
            min_eval = np.inf
            time_since_last_improvement = 0
            best_learner_state_dict = copy.deepcopy(self.learner.state_dict())
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, mode='min', factor=0.5,
                patience=5, threshold=0.0, threshold_mode='abs', cooldown=0, min_lr=0,
                eps=1e-08)

        for epoch in range(n_epochs):


            for it, (xb, yb) in enumerate(self.train_dl):

                self.learner.train()
                output = self.learner(xb)

                L1_reg_loss = 0.0
                if learner_l1 > 0.0:
                    L1_reg_loss = L1_reg(self.learner, learner_l1)

                D_loss = torch.mean((yb - output) ** 2) + L1_reg_loss

                self.optimizerD.zero_grad()
                D_loss.backward()
                self.optimizerD.step()
                self.learner.eval()

            if Xval is not None:  # if early stopping was enabled we check the out of sample violation

                output = self.learner(Xval)
                loss1 = np.mean(torch.mean((yval - output) ** 2).cpu().detach().numpy())
                self.curr_eval = loss1 

                lr_scheduler.step(self.curr_eval)

               
                if min_eval > self.curr_eval + earlystop_delta:
                    min_eval = self.curr_eval
                    time_since_last_improvement = 0
                    best_learner_state_dict = copy.deepcopy(
                        self.learner.state_dict())
                else:
                    time_since_last_improvement += 1
                    if time_since_last_improvement > earlystop_rounds:
                        break

            if self.logger is not None:
                self.logger(self, self.learner, epoch, self.writer)

        torch.save(self.learner, os.path.join(
            self.model_dir, "epoch{}".format(epoch)))

        self.n_epochs = epoch + 1
        if Xval is not None:
            self.learner.load_state_dict(best_learner_state_dict)
            torch.save(self.learner, os.path.join(
                self.model_dir, "earlystop"))

        return self

    def fit(self, X, y, Xval=None, yval=None, *,
            earlystop_rounds=20, earlystop_delta=0,
            learner_l2=1e-3, learner_l1=0, learner_lr=0.001,
            n_epochs=100, bs=100, optimizer='adam',
            warm_start=False, logger=None, model_dir='.', device=None):
        """
        Parameters
        ----------
        X : features of shape (n_samples, n_features)
        y : label of shape (n_samples, 1)
        Xval : validation set, if not None, then earlystopping is enabled based on out of sample moment violation
        yval : validation labels
        earlystop_rounds : how many epochs to wait for an out of sample improvement
        earlystop_delta : min increment for improvement for early stopping
        learner_l2 : l2_regularization of parameters of learner
        learner_l1 : l1_regularization of parameters of learner
        learner_lr : learning rate of the Adam optimizer for learner
        n_epochs : how many passes over the data
        bs : batch size
        target_reg : float in [0, 1]. weight on targeted regularization vs mse loss
        optimizer : one of {'adam', 'rmsprop', 'sgd'}. default='adam'
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        model_dir : folder where to store the learned models after every epoch
        device : name of device on which to perform all computation
        """

        X, y, Xval, yval = self._pretrain(X, y, Xval, yval, bs=bs, warm_start=warm_start,
                                 logger=logger, model_dir=model_dir,
                                 device=device)

        self._train(X, y, Xval=Xval, yval=yval,
                    earlystop_rounds=earlystop_rounds, earlystop_delta=earlystop_delta,
                    learner_l2=learner_l2, learner_l1=learner_l1,
                    learner_lr=learner_lr, n_epochs=n_epochs, bs=bs,
                    optimizer=optimizer)

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self

    def get_model(self, model):
        if model == 'final':
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(self.n_epochs - 1)),
                                            weights_only = False)
        if model == 'earlystop':
            return torch.load(os.path.join(self.model_dir,
                                           "earlystop"),
                                           weights_only = False)

        raise AttributeError("Not implemented")

    def predict(self, X, model='final'):
        """
        Parameters
        ----------
        X : (n, p) matrix of features
        model : one of ('final', 'earlystop'), whether to use an average of models or the final
        Returns
        -------
        ypred, apred : (n, 2) matrix of learned regression and riesz representers g(X), a(X)
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X).to(self.device)

        return self.get_model(model)(X).cpu().data

