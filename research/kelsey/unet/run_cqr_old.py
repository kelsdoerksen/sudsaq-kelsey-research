import torch
import sys
import os
from model import *
import numpy as np
from functools import partial
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/submodules/cqr')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd() + '/submodules/nonconformist')))

from nonconformist.base import RegressorAdapter
from cqr.torch_models import AllQuantileLoss
from nonconformist.nc import RegressorNc
from nonconformist.nc import QuantileRegErrFunc
from cqr import helper


# --- Helper Functions

def mask_nans(predictions, labels):
    """
    Mask NaNs for calculating loss function
    """

    # Apply mask to remove NaNs when making predictions/calculating loss
    preds_np = predictions.cpu().detach().numpy()
    preds_low = preds_np[:, 0, :, :]
    preds_high = preds_np[:, 0, :, :]

    mask = ~torch.isnan(labels)
    mask = mask.numpy()
    mask = mask[:, 0, :, :]

    preds_low = preds_low[mask]
    preds_high = preds_high[mask]

    batch_y_np = labels.numpy()
    batch_y_np = batch_y_np[:, 0, :, :]
    batch_y_np_mask = batch_y_np[mask]

    preds_combo = np.array([preds_low, preds_high])
    preds_combo = preds_combo.reshape((len(batch_y_np_mask), 2))

    # Back to torch for loss calc
    preds_combo_torch = torch.from_numpy(preds_combo).float().requires_grad_(True)
    batch_y_np_mask_torch = torch.from_numpy(batch_y_np_mask).float().requires_grad_(False)

    return preds_combo_torch, batch_y_np_mask_torch


def epoch_trainer(model, loss_func, x_train, y_train, batch_size, optimizer, cnt=0, best_cnt=np.Inf):
    """ Sweep over the data and update the model's parameters

    Parameters
    ----------

    model : class of neural net model
    loss_func : class of loss function
    x_train : pytorch tensor n training features, each of dimension p (nXp)
    batch_size : integer, size of the mini-batch
    optimizer : class of SGD solver
    cnt : integer, counting the gradient steps
    best_cnt: integer, stop the training if current cnt > best_cnt

    Returns
    -------

    epoch_loss : mean loss value
    cnt : integer, cumulative number of gradient steps

    """
    model.train()
    shuffle_idx = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle_idx)
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    epoch_losses = []
    for idx in range(0, x_train.shape[0], batch_size):
        cnt = cnt + 1
        optimizer.zero_grad()

        batch_x = x_train[idx: min(idx + batch_size, x_train.shape[0]),:]
        batch_y = y_train[idx: min(idx + batch_size, y_train.shape[0])]

        preds = model(batch_x)

        preds_masked, labels_masked = mask_nans(preds, batch_y)

        # Calculate loss with removed NaNs
        loss = loss_func(preds_masked, labels_masked)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.cpu().detach().numpy())

        if cnt >= best_cnt:
            break

    epoch_loss = np.mean(epoch_losses)

    return epoch_loss, cnt


def rearrange(all_quantiles, quantile_low, quantile_high, test_preds):
    """ Produce monotonic quantiles

    Parameters
    ----------

    all_quantiles : numpy array (q), grid of quantile levels in the range (0,1)
    quantile_low : float, desired low quantile in the range (0,1)
    quantile_high : float, desired high quantile in the range (0,1)
    test_preds : numpy array of predicted quantile (nXq)

    Returns
    -------

    q_fixed : numpy array (nX2), containing the rearranged estimates of the
              desired low and high quantile

    References
    ----------
    .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
            "Quantile and probability curves without crossing."
            Econometrica 78.3 (2010): 1093-1125.

    """
    scaling = all_quantiles[-1] - all_quantiles[0]
    low_val = (quantile_low - all_quantiles[0])/scaling
    high_val = (quantile_high - all_quantiles[0])/scaling
    q_fixed = np.quantile(test_preds,(low_val, high_val),interpolation='linear',axis=1)
    return q_fixed.T


# --- Conformal Quantile Refression Classes
class CQRLearnerOptimizedCrossing:
    """
    Fit conditional quantile UNet to training data
    """
    def __init__(self, model, optimizer_class, loss_func, device='cpu', test_ratio=0.2, random_state=0,
                 qlow=0.05, qhigh=0.95, use_rearrangement=False):
        """ Initialization

        Parameters
        ----------

        model : class of neural network model
        optimizer_class : class of SGD optimizer (e.g. pytorch's Adam)
        loss_func : loss to minimize
        device : string, "cuda:0" or "cpu"
        test_ratio : float, test size used in cross-validation (CV)
        random_state : integer, seed used in CV when splitting to train-test
        qlow : float, low quantile level in the range (0,1)
        qhigh : float, high quantile level in the range (0,1)
        use_rearrangement : boolean, use the rearrangement  algorithm (True)
                            of not (False)

        """
        self.model = model.to(device)
        self.use_rearrangement = use_rearrangement
        self.compute_coverage = True
        self.quantile_low = qlow
        self.quantile_high = qhigh
        self.target_coverage = 100.0*(self.quantile_high - self.quantile_low)
        self.all_quantiles = loss_func.quantiles
        self.optimizer_class = optimizer_class
        self.optimizer = optimizer_class(self.model.parameters())
        self.loss_func = loss_func.to(device)
        self.device = device
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []

    def fit(self, x, y, epochs, batch_size, verbose=False):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size used in SGD solver

        """
        sys.stdout.flush()
        device = 'cpu'
        model = copy.deepcopy(self.model)
        model = model.to(device)
        optimizer = self.optimizer_class(model.parameters())
        best_epoch = epochs

        x_train, xx, y_train, yy = train_test_split(x,
                                                    y,
                                                    test_size=self.test_ratio,
                                                    random_state=self.random_state)

        x_train.to(self.device).requires_grad_(False)
        xx.to(self.device).requires_grad_(False)
        y_train.to(self.device).requires_grad_(False)
        yy_cpu = yy
        yy.to(self.device).requires_grad_(False)

        best_avg_length = 1e10
        best_coverage = 0
        best_cnt = 1e10

        cnt = 0
        for e in range(epochs):
            model.train()
            epoch_loss, cnt = epoch_trainer(model, self.loss_func, x_train, y_train, batch_size, optimizer, cnt)
            self.loss_history.append(epoch_loss)

            model.eval()
            preds = model(xx)
            preds_no_nan, labels_no_nan = mask_nans(preds, yy)
            test_epoch_loss = self.loss_func(preds_no_nan, labels_no_nan).cpu().detach().numpy()
            self.test_loss_history.append(test_epoch_loss)

            test_preds = preds.cpu().detach().numpy()
            test_preds = np.squeeze(test_preds)

            if self.use_rearrangement:
                test_preds = rearrange(self.all_quantiles, self.quantile_low, self.quantile_high, test_preds)

            preds_no_nan = preds_no_nan.detach().numpy()
            y_lower = preds_no_nan[:,0]
            y_upper = preds_no_nan[:,1]
            yy_cpu_no_nan = labels_no_nan.numpy()
            coverage, avg_length = helper.compute_coverage_len(yy_cpu_no_nan, y_lower, y_upper)

            if (coverage >= self.target_coverage) and (avg_length < best_avg_length):
                best_avg_length = avg_length
                best_coverage = coverage
                best_epoch = e
                best_cnt = cnt


            print("CV: Epoch {}: Train {}, Test {}, Best epoch {}, Best Coverage {} Best Length {} Cur Coverage {}".format(e+1, epoch_loss, test_epoch_loss, best_epoch, best_coverage, best_avg_length, coverage))

        x = x.to(self.device).requires_grad_(False)
        y = y.to(self.device).requires_grad_(False)

        cnt = 0
        for e in range(best_epoch+1):
            if cnt > best_cnt:
                break
            epoch_loss, cnt = epoch_trainer(self.model, self.loss_func, x, y, batch_size, self.optimizer, cnt, best_cnt)
            self.full_loss_history.append(epoch_loss)

            print("Full: Epoch {}: {}, cnt {}".format(e+1, epoch_loss, cnt))

    def predict(self, x):
        """ Estimate the conditional low and high quantile given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        test_preds : numpy array of predicted low and high quantiles (nX2)

        """
        self.model.eval()
        device = 'cpu'
        test_preds = self.model(x.to(self.device).requires_grad_(False)).cpu().detach().numpy()
        if self.use_rearrangement:
            test_preds = rearrange(self.all_quantiles, self.quantile_low, self.quantile_high, test_preds)
        else:
            test_preds[:,0] = np.min(test_preds,axis=1)
            test_preds[:,1] = np.max(test_preds,axis=1)
        return test_preds


class UNet_RegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, formulated as UNet
    """
    def __init__(self,
                 model,
                 fit_params=None,
                 n_channels=32,
                 hidden_size=1,
                 quantiles=[.1, .9],
                 learn_func=torch.optim.Adam,
                 epochs=1000,
                 batch_size=10,
                 dropout=0.1,
                 lr=0.01,
                 wd=1e-6,
                 test_ratio=0.2,
                 random_state=0,
                 use_rearrangement=False):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        quantiles : numpy array, low and high quantile levels in range (0,1)
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation
        use_rearrangement : boolean, use the rearrangement algorithm (True)
                            of not (False). See reference [1].

        References
        ----------
        .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
                "Quantile and probability curves without crossing."
                Econometrica 78.3 (2010): 1093-1125.

        """
        super(UNet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        if use_rearrangement:
            self.all_quantiles = torch.from_numpy(np.linspace(0.01,0.99,99)).float()
        else:
            self.all_quantiles = self.quantiles
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = dropout
        self.lr = lr
        self.n_channels = n_channels
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.wd = wd
        self.learn_func = learn_func
        self.use_rearrangement = use_rearrangement
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.model = models.CQRUNet(quantiles=self.all_quantiles,
                                              n_channels=n_channels)
        self.loss_func = AllQuantileLoss(self.all_quantiles)
        self.learner = CQRLearnerOptimizedCrossing(self.model,
                                                partial(learn_func, lr=lr, weight_decay=wd),
                                                self.loss_func,
                                                device=self.device,
                                                test_ratio=self.test_ratio,
                                                random_state=self.random_state,
                                                qlow=self.quantiles[0],
                                                qhigh=self.quantiles[1],
                                                use_rearrangement=use_rearrangement)
        device='cpu'
        self.model.to(device=device)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(x, y, self.epochs, self.batch_size)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        return self.learner.predict(x)


def run_cqr_pipeline(n_channels,
                     quantiles,
                     epochs,
                     batch_size,
                     lr,
                     wd,
                     random_state,
                     train_dataset,
                     test_dataset,
                     alpha,
                     test_ratio=None,
                     use_rearrangement=False
                     ):
    """
    Run pipeline to generate upper and lower
    predictive bounds and compute average coverage
    and average length
    """
    quantile_estimator = UNet_RegressorAdapter(model=None,
                                               fit_params=None,
                                               n_channels=n_channels,
                                               hidden_size=None,
                                               quantiles=quantiles,
                                               learn_func=torch.optim.Adam,
                                               epochs=epochs,
                                               batch_size=batch_size,
                                               dropout=None,
                                               lr=lr,
                                               wd=wd,
                                               test_ratio=test_ratio,
                                               random_state=random_state,
                                               use_rearrangement=use_rearrangement)

    nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

    # --- DataLoaders
    # The DataLoader pulls instances of data from the Dataset, collects them in batches,
    # and returns them for consumption by your training loop.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xtrain, ytrain, xtest, ytest = [], [], [], []
    # Iterate through train, test and get in format xtrain, ytrain, xtest, ytest
    for i, data in enumerate(train_loader):
        input, label = data
        input, label = input.to(device), label.to(device)
        xtrain.append(input)
        ytrain.append(label)

    for i, data in enumerate(test_loader):
        test_input, test_label = data
        test_input, test_label = test_input.to(device), test_label.to(device)
        xtest.append(input)
        ytest.append(label)

    # Split data into train and calibration
    n_train = len(train_loader)
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2 * n_half]

    # Compute CQR bounds
    y_lower, y_upper = helper.run_icp(nc, xtrain[0], ytrain[0], xtest[0], ytest[0], idx_train, idx_cal, alpha)

    # Compute and print average coverage and average length
    coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(ytest,
                                                               y_lower,
                                                               y_upper,
                                                               alpha,
                                                               "CQR Neural Net")
