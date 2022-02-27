from __future__ import annotations
import math
import torch
import gpytorch
# import xlwt

import torch

from gpytorch.kernels import RBFKernel
import xlrd
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import squareform
import xlwt
import scipy.cluster.hierarchy as spc





import time
import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim.numpy_converter import (
    TorchAttr,
    module_to_array,
    set_params_with_array,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import (
    _filter_kwargs,
    _get_extra_mll_args,
    _scipy_objective_and_grad,
)
from gpytorch import settings as gpt_settings
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from scipy.optimize import Bounds, minimize
from torch import Tensor
from torch.nn import Module
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.optimizer import Optimizer


ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]
TScipyObjective = Callable[
    [np.ndarray, MarginalLogLikelihood, Dict[str, TorchAttr]], Tuple[float, np.ndarray]
]
TModToArray = Callable[
    [Module, Optional[ParameterBounds], Optional[Set[str]]],
    Tuple[np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]],
]
TArrayToMod = Callable[[Module, np.ndarray, Dict[str, TorchAttr]], Module]


class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


def fit_gpytorch_torch(
    projection,
    projection_len,
    ard_len,
    mll: MarginalLogLikelihood,
    gamma=0.002,
    bounds: Optional[ParameterBounds] = None,
    optimizer_cls: Optimizer = Adam,
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = True,
    approx_mll: bool = True,
) -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationIteration]]]]:
    r"""Fit a gpytorch model by maximizing MLL with a torch optimizer.

    The model and likelihood in mll must already be in train mode.
    Note: this method requires that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        bounds: A ParameterBounds dictionary mapping parameter names to tuples
            of lower and upper bounds. Bounds specified here take precedence
            over bounds on the same parameters specified in the constraints
            registered with the module.
        optimizer_cls: Torch optimizer to use. Must not require a closure.
        options: options for model fitting. Relevant options will be passed to
            the `optimizer_cls`. Additionally, options can include: "disp"
            to specify whether to display model fitting diagnostics and "maxiter"
            to specify the maximum number of iterations.
        track_iterations: Track the function values and wall time for each
            iteration.
        approx_mll: If True, use gpytorch's approximate MLL computation (
            according to the gpytorch defaults based on the training at size).
            Unlike for the deterministic algorithms used in fit_gpytorch_scipy,
            this is not an issue for stochastic optimizers.

    Returns:
        2-element tuple containing
        - mll with parameters optimized in-place.
        - Dictionary with the following key/values:
        "fopt": Best mll value.
        "wall_time": Wall time of fitting.
        "iterations": List of OptimizationIteration objects with information on each
        iteration. If track_iterations is False, will be empty.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> mll.train()
        >>> fit_gpytorch_torch(mll)
        >>> mll.eval()
    """
    optim_options = {"maxiter": 100, "disp": True, "lr": 0.05}
    optim_options.update(options or {})
    exclude = optim_options.pop("exclude", None)
    if exclude is not None:
        mll_params = [
            t for p_name, t in mll.named_parameters() if p_name not in exclude
        ]
    else:
        mll_params = list(mll.parameters())
    optimizer = optimizer_cls(
        params=[{"params": mll_params}],
        **_filter_kwargs(optimizer_cls, **optim_options),
    )

    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(mll, "named_parameters_and_constraints"):
        for param_name, _, constraint in mll.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    iterations = []
    t1 = time.time()

    param_trajectory: Dict[str, List[Tensor]] = {
        name: [] for name, param in mll.named_parameters()
    }
    loss_trajectory: List[float] = []
    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **optim_options)
    )
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    while not stop:
        optimizer.zero_grad()
        with gpt_settings.fast_computations(log_prob=approx_mll):
            output = mll.model(*train_inputs)
            # we sum here to support batch mode
            args = [output, train_targets] + _get_extra_mll_args(mll)

            estimated_covar = projection / (
                projection_len ** 2) @ projection.t() + \
                torch.diag(torch.squeeze(ard_len.reciprocal() ** 2))

            covar_inv_diags = estimated_covar.diag()  # ** 0.5
            estimated_corr = estimated_covar / torch.outer(covar_inv_diags ** 0.5, covar_inv_diags ** 0.5)

            l1_loss=estimated_corr.abs().sum()*gamma

            loss = -mll(*args).sum()+l1_loss
            loss.backward(retain_graph=True)
        loss_trajectory.append(loss.item())
        for name, param in mll.named_parameters():
            param_trajectory[name].append(param.detach().clone())
        if optim_options["disp"] and (
            (i + 1) % 10 == 0 or i == (optim_options["maxiter"] - 1)
        ):
            print(f"Iter {i + 1}/{optim_options['maxiter']}: {loss.item()}")
        if track_iterations:
            iterations.append(OptimizationIteration(i, loss.item(), time.time() - t1))
        optimizer.step()
        # project onto bounds:
        if bounds_:
            for pname, param in mll.named_parameters():
                if pname in bounds_:
                    param.data = param.data.clamp(*bounds_[pname])
        i += 1
        stop = stopping_criterion.evaluate(fvals=loss.detach())
    info_dict = {
        "fopt": loss_trajectory[-1],
        "wall_time": time.time() - t1,
        "iterations": iterations,
    }
    return mll, info_dict


def generate_suredata(dim, file_loc=r"16Mdata1.xls"):
    # X data generation
    excel = xlrd.open_workbook(file_loc)

    all_sheet = excel.sheets()

    database = all_sheet[0]
    result0 = database.col_values(0)[3:]
    matname0 = database.col_values(1)[3:]
    dsq0 = database.col_values(2)[3:]
    dv0 = database.col_values(3)[3:]
    sqatom0 = database.col_values(8)[3:]
    for i in range(len(sqatom0)):
        if sqatom0[i]=='D':
            sqatom0[i]='H'
    adata0 = database.col_values(10)[3:]
    cdata0 = database.col_values(11)[3:]
    scdata0 = database.col_values(12)[3:]
    result=[]
    matname=[]
    dsq=[]
    dv=[]
    sqatom=[]
    adata=[]
    cdata=[]
    # add
    for i in range(len(matname0)):
        if result0[i][-1]!='*':
            result.append(result0[i])
            matname.append(matname0[i])
            dsq.append(dsq0[i])
            dv.append(dv0[i])
            sqatom.append(sqatom0[i])
            adata.append(adata0[i])
            cdata.append(cdata0[i])
    excel1 = xlrd.open_workbook(r"atomic.xls")
    all_sheet1 = excel1.sheets()
    atomicdata = all_sheet1[0]
    ele1 = atomicdata.col_values(0)
    elene1 = atomicdata.col_values(1)
    elena1 = atomicdata.col_values(2)
    eleip1 = atomicdata.col_values(3)
    rcov1 = atomicdata.col_values(4)
    excel2 = xlrd.open_workbook(r"xenonpy.xls")
    all_sheet2 = excel2.sheets()
    xenondata = all_sheet2[0]
    polar = xenondata.col_values(57)
    fcc = xenondata.col_values(25)
    excel3 = xlrd.open_workbook(r"Econfig.xls")
    all_sheet3 = excel3.sheets()
    edata = all_sheet3[0]
    ele2 = edata.col_values(0)[1:]
    valence = edata.col_values(1)[1:]

    Xtot = []
    Ytot = []
    composition = []
    elist = []
    labels = [x[0] == "y" for x in result]
    for i in range(len(matname)):
        elements = []
        ratio = []
        for j in range(len(matname[i])):
            if (j < len(matname[i]) - 1) and (matname[i][j] == r')') and (ord(matname[i][j + 1]) > 47) and (
                    ord(matname[i][j + 1]) < 58):
                print(i, matname[i])
            if (ord(matname[i][j]) > 64) & (ord(matname[i][j]) < 91):
                ele = matname[i][j]
                if j == (len(matname[i]) - 1):
                    rate = 1.
                if j < (len(matname[i]) - 1):
                    if (ord(matname[i][j + 1]) > 96) & (ord(matname[i][j + 1]) < 123):
                        ele = ele + matname[i][j + 1]
                        if (j + 2 == len(matname[i])):
                            rate = 1.
                        elif ((ord(matname[i][j + 2]) > 64) & (ord(matname[i][j + 2]) < 91)) or (
                                matname[i][j + 2] == r' ') or (matname[i][j + 2] == r'(') or (
                                matname[i][j + 2] == r')'):
                            rate = 1.
                        else:
                            step = 2
                            string = ''
                            while (j + step < len(matname[i])) and ((ord(matname[i][j + step]) == 46) or (
                                    (ord(matname[i][j + step]) > 47) and (ord(matname[i][j + step]) < 58))):
                                string = string + matname[i][j + step]
                                step = step + 1
                            rate = float(string)
                    elif ((ord(matname[i][j + 1]) > 64) & (ord(matname[i][j + 1]) < 91)) or (
                            matname[i][j + 1] == r' ') or (matname[i][j + 1] == r'(') or (matname[i][j + 1] == r')'):
                        rate = 1.
                    else:
                        step = 1
                        string = ''
                        while (j + step < len(matname[i])) and ((ord(matname[i][j + step]) == 46) or (
                                (ord(matname[i][j + step]) > 47) and (ord(matname[i][j + step]) < 58))):
                            string = string + matname[i][j + step]
                            step = step + 1
                        rate = float(string)
                rep = 0
                for k in range(len(elements)):
                    if (elements[k] == ele):
                        rep = 1
                        ratio[k] = ratio[k] + rate
                if (rep == 0):
                    elements.append(ele)
                    ratio.append(rate)
        composition.append(ratio)
        elist.append(elements)
        ea, ip, en, rc, ve, pl = [], [], [], [], [], []
        tv = 0.
        for j in range(len(elements)):
            for k in range(len(ele2)):
                if elements[j] == ele2[k]:
                    tv = tv + float(ratio[j]) * float(valence[k])

        for j in range(len(elements)):
            for k in range(len(ele2)):
                if elements[j] == ele2[k]:
                    pl.append(polar[k])
                    ve.append(valence[k])

        for j in range(len(elements)):
            for k in range(len(ele1)):
                if (elements[j] == ele1[k]):
                    ip.append(eleip1[k])
                    ea.append(elena1[k])
                    en.append(elene1[k])
                    rc.append(rcov1[k])


        for k in range(len(ele1)):
            if sqatom[i] == ele1[k]:
                ipsq = eleip1[k]
                easq = elena1[k]
                ensq = elene1[k]
                rcsq = rcov1[k]

        for k in range(len(ele2)):
            if sqatom[i] == ele2[k]:
                plsq = polar[k]
                vesq = valence[k]
                fccsq = fcc[k]

        datapoint = [max(rc), min(ea), easq, min(en), ensq, max(pl), min(pl), plsq, max(ve), min(ve), vesq, tv]
        for j in range(12-dim):
            del(datapoint[0])
        if (dsq[i] > 2.25) and (dsq[i] < 3.75) and (dv[i] < 3.5) and (cdata[i] < 15.) and (max(rc) < 220) and (
                min(rc) > 40):
            Xtot.append(datapoint)  # eamin, enmin, ipmin,    , rmin
            Ytot.append(labels[i])
    print(len(Xtot), len(Xtot[0]))
    Xraw = copy.deepcopy(Xtot)
    for i in range(len(Xtot[0])):
        max1 = Xtot[0][i]
        min1 = Xtot[0][i]
        for j in range(len(Xtot)):
            if Xtot[j][i] > max1:
                max1 = Xtot[j][i]
            if Xtot[j][i] < min1:
                min1 = Xtot[j][i]
        ruler = max1 - min1
        for j in range(len(Xtot)):
            Xtot[j][i] = Xtot[j][i] - min1
            Xtot[j][i] = Xtot[j][i] / ruler

    Ytot = torch.tensor(Ytot) * 1
    count = 0
    X1 = []
    X0 = []
    for i in range(len(Ytot)):
        if Ytot[i] == 1:
            X1.append(Xtot[i])
        else:
            X0.append(Xtot[i])
    print(count)
    return torch.tensor(X1, dtype=torch.float64), torch.tensor(X0, dtype=torch.float64), Ytot


def generate_guessdata(dim, file_loc=r"16Mdata1.xls"):
    # X data generation
    excel = xlrd.open_workbook(file_loc)

    all_sheet = excel.sheets()

    database = all_sheet[0]
    result0 = database.col_values(0)[3:]
    matname0 = database.col_values(1)[3:]
    dsq0 = database.col_values(2)[3:]
    dv0 = database.col_values(3)[3:]
    sqatom0 = database.col_values(8)[3:]
    for i in range(len(sqatom0)):
        if sqatom0[i] == 'D':
            sqatom0[i] = 'H'
    adata0 = database.col_values(10)[3:]
    cdata0 = database.col_values(11)[3:]
    scdata0 = database.col_values(12)[3:]
    result = []
    matname = []
    dsq = []
    dv = []
    sqatom = []
    adata = []
    cdata = []
    # add
    for i in range(len(matname0)):
        if result0[i][-1] == '*':
            result.append(result0[i])
            matname.append(matname0[i])
            dsq.append(dsq0[i])
            dv.append(dv0[i])
            sqatom.append(sqatom0[i])
            adata.append(adata0[i])
            cdata.append(cdata0[i])
    excel1 = xlrd.open_workbook(r"atomic.xls")
    all_sheet1 = excel1.sheets()
    atomicdata = all_sheet1[0]
    ele1 = atomicdata.col_values(0)
    elene1 = atomicdata.col_values(1)
    elena1 = atomicdata.col_values(2)
    eleip1 = atomicdata.col_values(3)
    rcov1 = atomicdata.col_values(4)
    excel2 = xlrd.open_workbook(r"xenonpy.xls")
    all_sheet2 = excel2.sheets()
    xenondata = all_sheet2[0]
    polar = xenondata.col_values(57)
    fcc = xenondata.col_values(25)
    excel3 = xlrd.open_workbook(r"Econfig.xls")
    all_sheet3 = excel3.sheets()
    edata = all_sheet3[0]
    ele2 = edata.col_values(0)[1:]
    valence = edata.col_values(1)[1:]

    Xtot = []
    Ytot = []
    gmat = []
    composition = []
    elist = []
    labels = [x[0] == "y" for x in result]
    for i in range(len(matname)):
        elements = []
        ratio = []
        for j in range(len(matname[i])):
            if (j < len(matname[i]) - 1) and (matname[i][j] == r')') and (ord(matname[i][j + 1]) > 47) and (
                    ord(matname[i][j + 1]) < 58):
                print(i, matname[i])
            if (ord(matname[i][j]) > 64) & (ord(matname[i][j]) < 91):
                ele = matname[i][j]
                if j == (len(matname[i]) - 1):
                    rate = 1.
                if j < (len(matname[i]) - 1):
                    if (ord(matname[i][j + 1]) > 96) & (ord(matname[i][j + 1]) < 123):
                        ele = ele + matname[i][j + 1]
                        if (j + 2 == len(matname[i])):
                            rate = 1.
                        elif ((ord(matname[i][j + 2]) > 64) & (ord(matname[i][j + 2]) < 91)) or (
                                matname[i][j + 2] == r' ') or (matname[i][j + 2] == r'(') or (
                                matname[i][j + 2] == r')'):
                            rate = 1.
                        else:
                            step = 2
                            string = ''
                            while (j + step < len(matname[i])) and ((ord(matname[i][j + step]) == 46) or (
                                    (ord(matname[i][j + step]) > 47) and (ord(matname[i][j + step]) < 58))):
                                string = string + matname[i][j + step]
                                step = step + 1
                            rate = float(string)
                    elif ((ord(matname[i][j + 1]) > 64) & (ord(matname[i][j + 1]) < 91)) or (
                            matname[i][j + 1] == r' ') or (matname[i][j + 1] == r'(') or (matname[i][j + 1] == r')'):
                        rate = 1.
                    else:
                        step = 1
                        string = ''
                        while (j + step < len(matname[i])) and ((ord(matname[i][j + step]) == 46) or (
                                (ord(matname[i][j + step]) > 47) and (ord(matname[i][j + step]) < 58))):
                            string = string + matname[i][j + step]
                            step = step + 1
                        rate = float(string)
                rep = 0
                for k in range(len(elements)):
                    if (elements[k] == ele):
                        rep = 1
                        ratio[k] = ratio[k] + rate
                if (rep == 0):
                    elements.append(ele)
                    ratio.append(rate)
        composition.append(ratio)
        elist.append(elements)
        ea, ip, en, rc, ve, pl = [], [], [], [], [], []
        tv = 0.
        for j in range(len(elements)):
            for k in range(len(ele2)):
                if elements[j] == ele2[k]:
                    tv = tv + float(ratio[j]) * float(valence[k])
        for j in range(len(elements)):
            for k in range(len(ele2)):
                if elements[j] == ele2[k]:
                    pl.append(polar[k])
                    ve.append(valence[k])

        for j in range(len(elements)):
            for k in range(len(ele1)):
                if (elements[j] == ele1[k]):
                    ip.append(eleip1[k])
                    ea.append(elena1[k])
                    en.append(elene1[k])
                    rc.append(rcov1[k])

        for k in range(len(ele1)):
            if sqatom[i] == ele1[k]:
                ipsq = eleip1[k]
                easq = elena1[k]
                ensq = elene1[k]
                rcsq = rcov1[k]

        for k in range(len(ele2)):
            if sqatom[i] == ele2[k]:
                plsq = polar[k]
                vesq = valence[k]
                fccsq = fcc[k]

        datapoint = [max(rc), min(ea), easq, min(en), ensq, max(pl), min(pl), plsq, max(ve), min(ve), vesq, tv]
        for k in range(12-dim):
            del(datapoint[0])
        if (dsq[i] > 2.25) and (dsq[i] < 3.75) and (dv[i] < 3.5) and (cdata[i] < 15.) and (max(rc) < 220) and (
                min(rc) > 40):
            Xtot.append(datapoint)  # eamin, enmin, ipmin,    , rmin
            Ytot.append(labels[i])
            gmat.append(matname[i])
    print(len(Xtot), len(Xtot[0]))
    Xraw = copy.deepcopy(Xtot)
    for i in range(len(Xtot[0])):
        max1 = Xtot[0][i]
        min1 = Xtot[0][i]
        for j in range(len(Xtot)):
            if Xtot[j][i] > max1:
                max1 = Xtot[j][i]
            if Xtot[j][i] < min1:
                min1 = Xtot[j][i]
        ruler = max1 - min1
        for j in range(len(Xtot)):
            Xtot[j][i] = Xtot[j][i] - min1
            Xtot[j][i] = Xtot[j][i] / ruler

    Ytot = torch.tensor(Ytot) * 1
    print(len(Xtot), len(Ytot))
    return gmat, torch.tensor(Xtot, dtype=torch.float32), Ytot



device = torch.device("cuda:0")
dtype = torch.double

class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, initialization, rank=3, interval1=10.):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module_projection = RBFKernel(
            ard_num_dims=rank,
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1),
            # lengthscale_prior=gpytorch.priors
        )
        #proj = torch.tensor(initialization, dtype=torch.float32)
        proj = torch.rand(train_x.shape[-1], rank).to(train_x)
        proj /= (proj ** 2).sum()
        proj.detach_().requires_grad_()
        self.register_parameter(
            "projection", torch.nn.Parameter(proj)
        )
        self.covar_module_ard = RBFKernel(
            ard_num_dims=train_x.shape[-1],
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1)
        )

    def forward(self, x):
        proj_x = x.matmul(self.projection)

        mean_x = self.mean_module(x)

        # this kernel is exp(-l_1^2 (x - x')P P^T(x - x') - l_2^2 (x - x')D(x - x'))
        # because we compute the product elementwise
        covar_x = self.covar_module_projection(proj_x) * self.covar_module_ard(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def make_and_fit_classifier(train_x, train_y, inbuffer, maxiter=2000, lr=0.001, rank=3, interval1=10.):

    likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
    model = DirichletGPModel(
        train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes, initialization=inbuffer,
        rank=rank, interval1=interval1
    )
    model = model.to(train_x)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _, info_dict = fit_gpytorch_torch(model.projection, model.covar_module_projection.lengthscale, model.covar_module_ard.lengthscale, mll, options={"maxiter": 1000, "lr": lr})
    print("Final MLL: ", info_dict["fopt"])

    return model, info_dict["fopt"]


def compute_accuracy(model, likelihood, test_x, test_y, train_x, train_y, gx, gy, matname, collect):
    likelihood.eval()
    model.eval()

    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(test_x)

        pred_means = test_dist.loc

        y_pred = []

        for i in range(len(pred_means[0])):
            if pred_means[0][i] > pred_means[1][i]:
                y_pred.append(0.)
            else:
                y_pred.append(1.)
    count = 0.0
    count0 = 0.0
    count1 = 0.0
    tot0=0.
    tot1=0.
    testy = test_y.float()
    for i in range(len(testy)):
        if testy[i]==1:
            tot1=tot1+1.
        elif testy[i]==0.:
            tot0=tot0+1.
        if y_pred[i] == testy[i]:
            count = count + 1.
            if y_pred[i] == 1.:
                count1=count1+1
            else:
                count0 = count0+1
    acc0 = count0 / tot0
    acc1 = count1 / tot1
    acct = (count1 + count0) / (tot1 + tot0)


    with gpytorch.settings.fast_pred_var(), torch.no_grad():

        test_dist = model(gx)

        pred_means = test_dist.loc

        y_pred = []

        for i in range(len(pred_means[0])):
            if pred_means[0][i] > pred_means[1][i]:
                y_pred.append(0.)
            else:
                y_pred.append(1.)
    count = 0.0
    count0 = 0.0
    count1 = 0.0
    tot0 = 0.
    tot1 = 0.
    testy = gy.float()
    for i in range(len(testy)):
        if testy[i]==1:
            tot1=tot1+1.
        elif testy[i]==0.:
            tot0=tot0+1.
        if y_pred[i] == testy[i]:
            count = count + 1.
            if y_pred[i] == 1.:
                count1=count1+1
            else:
                count0 = count0+1
        else:
            sign=0
            for j in range(len(collect)):
                if collect[j]==matname[i]:
                    sign=1
            if sign==0:
                collect.append(matname[i])
    acg0 = count0 / tot0
    acg1 = count1 / tot1

    with torch.no_grad():
        test_dist2 = model(train_x)

        pred_means = test_dist2.loc

        y_pred2 = []

        for i in range(len(pred_means[0])):
            if pred_means[0][i] > pred_means[1][i]:
                y_pred2.append(0.)
            else:
                y_pred2.append(1.)
    count2 = 0.0
    testy2 = train_y.float()
    for i in range(len(testy2)):
        if y_pred2[i] == testy2[i]:
            count2 = count2 + 1.
    accuracy2 = count2 / train_x.shape[0]
    return acct, accuracy2, acc0, acc1, acg0, acg1, collect


def cluster_lengthscales(model, thresh_pct=0.15):
    # construct learned covariance matrix
    print(model.projection.cpu().data)
    print(model.covar_module_projection.lengthscale.cpu().data)
    print(model.covar_module_ard.lengthscale.cpu().data)
    cvector = model.projection.cpu().data.div(model.covar_module_projection.lengthscale.cpu().data)
    estimated_covar = model.projection / (model.covar_module_projection.lengthscale ** 2) @ model.projection.t() + \
                      torch.diag(torch.squeeze(model.covar_module_ard.lengthscale.reciprocal() ** 2))
    # buffer=torch.zeros([10,10], dtype=torch.float)
    # buffer=buffer.to(device)
    # print(torch.diag(model.covar_module_ard.lengthscale.reciprocal()).shape)
    # print(torch.diag(torch.squeeze(model.covar_module_ard.lengthscale.reciprocal())).cpu().data)
    # estimated_covar1= buffer+torch.diag(torch.squeeze(model.covar_module_ard.lengthscale.reciprocal()))
    # print(estimated_covar1)
    # construct learned correlation matrix
    covar_inv_diags = estimated_covar.diag()  # ** 0.5
    estimated_corr = estimated_covar / torch.outer(covar_inv_diags ** 0.5, covar_inv_diags ** 0.5)

    estimated_dist = covar_inv_diags.unsqueeze(0) + covar_inv_diags.unsqueeze(1) - 2. * estimated_covar
    # estimated_dist = 2. - 2. * estimated_corr

    # plot learned covariance matrix
   # fig, ax = plt.subplots(1, 2, figsize=(16, 5))
   # f = ax[0].imshow(estimated_covar.data.cpu().numpy())
   # fig.colorbar(f, ax=ax[0])
    #f = ax[1].imshow(estimated_corr.data.cpu().numpy())
    #fig.colorbar(f, ax=ax[1])
    #plt.show()

    # estimated_corr = estimated_corr.detach().cpu().numpy()
    # force symmetrize to prevent numerical instability

    return estimated_covar, estimated_corr, cvector  # , estimated_covar1


print(torch.__version__)
basis=torch.randn(12, 3, dtype=torch.float32)/3.
for d in range(12,11,-1):
    X1, X0, true_y = generate_suredata(dim=d)
    gmat, gDptt, g_y= generate_guessdata(dim=d)
    gDptt=gDptt.to(device)
    g_y=g_y.to(device)
    X1=X1.to(device)
    X0=X0.to(device)
    true_y = true_y.to(device)
    true_yf = true_y.float()
    inbuffer=[]
    for i in range(d):
        inbuffer.append([])
        for j in range(len(basis[i+12-d])):
            inbuffer[i].append(basis[i+12-d][j])

    cluster_list, acc_list, acc_list2, state_dict_list, mll_list, acc0_list, acc1_list, acg0_list,acg1_list, collect = [], [], [], [], [], [], [], [], [], []
    summatrix = torch.zeros([d, d], dtype=torch.float)
    summatrix = summatrix.to(device)
    rawmatrix = torch.zeros([d, d], dtype=torch.float)
    rawmatrix = rawmatrix.to(device)
    cvectors = torch.zeros([d, 3], dtype=torch.float)
    r1 = torch.zeros([d, d], dtype=torch.float)
    r1 = rawmatrix.to(device)
    for i in range(30):
        print("trying seed: ", i)
        torch.random.manual_seed(i+141)

        # note that now the training set is not fixed
        # lets use 80%
        shuffled_inds0 = torch.randperm(X0.shape[0])
        shuffled_inds1 = torch.randperm(X1.shape[0])
        trainset0 = shuffled_inds0[:241]
        testset0 = shuffled_inds0[241:]
        trainset1 = shuffled_inds1[:56]
        testset1 = shuffled_inds1[56:]

        train_x=torch.cat((X0[trainset0],X1[trainset1]),0)
        test_x=torch.cat((X0[testset0],X1[testset1]),0)
        y=[]
        for j in range(241):
            y.append(0)
        for j in range(56):
            y.append(1)
        train_y=torch.tensor(y)
        yt=[]
        for j in range(60):
            yt.append(0.)
        for j in range(14):
            yt.append(1.)
        test_y=torch.tensor(yt,dtype=torch.float64)
        with gpytorch.settings.max_cholesky_size(2000):
            model, mll1 = make_and_fit_classifier(train_x, train_y, inbuffer=inbuffer,
                                                  lr=0.1)
            acc, acc2, acc0, acc1, acg0, acg1, collect = compute_accuracy(model, model.likelihood, test_x, test_y,
                                         train_x, train_y, gDptt, g_y, gmat, collect)
            rmatrix, smatrix, cvector1 = cluster_lengthscales(model)
        summatrix = summatrix + smatrix
        rawmatrix = rawmatrix + rmatrix
        cvectors = cvectors + cvector1
        # r1=r1+r11
        acc_list.append(acc)
        acc_list2.append(acc2)
        acc0_list.append(acc0)
        acc1_list.append(acc1)
        acg0_list.append(acg0)
        acg1_list.append(acg1)
        mll_list.append(mll1)
        state_dict_list.append(model.state_dict)

    summatrix1 = summatrix.cpu()
    for i in range(len(summatrix1)):
        for j in range(len(summatrix1)):
            if j <= i:
                summatrix1[i][j] = 0.
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    f = plt.imshow(summatrix1.data.div(30.), cmap=mpl.cm.bwr, vmin=-1., vmax=1.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.gca().invert_yaxis()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ['fcsq', 'ea2', 'ea3', 'en2', 'en3', 'pl1', 'pl2', 'pl3', 've1', 've2', 've3', 'tv'], rotation=0)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ['fcsq', 'ea2', 'ea3', 'en2', 'en3', 'pl1', 'pl2', 'pl3', 've1', 've2', 've3', 'tv'], rotation=0)
    plt.colorbar(fraction=0.045)
    plt.savefig(str(d)+"M_z.png", bbox_inches="tight",transparent='true')

    rawmatrix1 = rawmatrix.cpu()
    maxraw=0.
    for i in range(len(rawmatrix1)):
        for j in range(len(rawmatrix1[i])):
            if abs(rawmatrix1[i][j])>maxraw:
                maxraw=rawmatrix1[i][j].data
    maxraw=maxraw/30.
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    f = plt.imshow(rawmatrix1.data.div(30.), cmap=mpl.cm.bwr,vmax=maxraw,vmin=-maxraw)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.gca().invert_yaxis()
    plt.colorbar(fraction=0.045)
    plt.savefig(str(d)+"LL+A.png", bbox_inches="tight",transparent='true')

    workbook=xlwt.Workbook(encoding='utf-8')
    worksheet=workbook.add_sheet('sheet1')
    for i in range(len(mll_list)):
        worksheet.write(0, i, label=mll_list[i])
        worksheet.write(2, i, label=acc_list[i])
        worksheet.write(4, i, label=acc_list2[i])
    worksheet.write(1,0,label=np.mean(mll_list))
    worksheet.write(1,2,label=np.std(mll_list))
    worksheet.write(3,0,label=np.mean(acc_list))
    worksheet.write(3,2,label=np.std(acc_list))
    worksheet.write(5,0,label=np.mean(acc_list2))
    worksheet.write(5,2,label=np.std(acc_list2))
    worksheet.write(7,0,label=np.mean(acc0_list))
    worksheet.write(7,2,label=np.mean(acc1_list))
    worksheet.write(8,0,label=np.mean(acg0_list))
    worksheet.write(8,2,label=np.mean(acg1_list))
    workbook.save(str(d)+'mll.xls')
    goodpo = []
    countpo = []
    goodne = []
    countne = []
    for i in range(12):
        for j in range(i + 1, 12):
            if summatrix1.data.div(30.).numpy()[i][j] > 0.:
                goodpo.append(summatrix1.data.div(30.).numpy()[i][j])
                countpo.append([i + 1, j + 1])
            elif summatrix1.data.div(30.).numpy()[i][j] < 0.:
                goodne.append(summatrix1.data.div(30.).numpy()[i][j])
                countne.append([i + 1, j + 1])
    for i in range(len(goodpo)):
        maxx = goodpo[i]
        place = i
        for j in range(i + 1, len(goodpo)):
            if abs(goodpo[j]) > abs(maxx):
                maxx = goodpo[j]
                place = j
        goodpo[place] = goodpo[i]
        goodpo[i] = maxx
        tt = countpo[place]
        countpo[place] = countpo[i]
        countpo[i] = tt

    for i in range(len(goodne)):
        maxx = goodne[i]
        place = i
        for j in range(i + 1, len(goodne)):
            if abs(goodne[j]) > abs(maxx):
                maxx = goodne[j]
                place = j
        goodne[place] = goodne[i]
        goodne[i] = maxx
        tt = countne[place]
        countne[place] = countne[i]
        countne[i] = tt
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('sheet1')
    for i in range(15):
        print(goodpo[i], countpo[i])
        worksheet.write(0, i, label=goodpo[i])
        worksheet.write(1, i, label=float(countpo[i][0]))
        worksheet.write(2, i, label=float(countpo[i][1]))

    for i in range(15):
        worksheet.write(4, i, label=goodne[i])
        worksheet.write(5, i, label=float(countne[i][0]))
        worksheet.write(6, i, label=float(countne[i][1]))
    workbook.save(str(d)+'order.xls')

    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('sheet1')
    diag = []
    order = []
    for i in range(len(rawmatrix1)):
        diag.append(rawmatrix1[i][i])
        order.append(i)
    for i in range(len(diag)):
        maxx = diag[i]
        place = i
        for j in range(i + 1, len(diag)):
            if abs(diag[j]) > abs(maxx):
                maxx = diag[j]
                place = j
        diag[place] = diag[i]
        diag[i] = maxx
        tt = order[place]
        order[place] = order[i]
        order[i] = tt
    for i in range(10):
        worksheet.write(0, i, label=float(order[i]))
    workbook.save(str(d)+'diag.xls')