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



def generate_suredata(dim, file_loc=r"C:\Users\ASUS\Desktop\TSM_document\16Mdata1.xls"):
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
    excel1 = xlrd.open_workbook(r"C:\Users\ASUS\Desktop\TSM_document\atomic.xls")
    all_sheet1 = excel1.sheets()
    atomicdata = all_sheet1[0]
    ele1 = atomicdata.col_values(0)
    elene1 = atomicdata.col_values(1)
    elena1 = atomicdata.col_values(2)
    eleip1 = atomicdata.col_values(3)
    rcov1 = atomicdata.col_values(4)
    excel2 = xlrd.open_workbook(r"C:\Users\ASUS\Desktop\TSM_document\xenonpy.xls")
    all_sheet2 = excel2.sheets()
    xenondata = all_sheet2[0]
    polar = xenondata.col_values(57)
    fcc = xenondata.col_values(25)
    excel3 = xlrd.open_workbook(r"C:\Users\ASUS\Desktop\TSM_document\Econfig.xls")
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

        datapoint = [fccsq, max(rc), min(ea), easq, min(en), ensq, max(pl), min(pl), plsq, max(ve), min(ve), vesq, tv]
        for j in range(13-dim):
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


def generate_guessdata(dim, file_loc=r"C:\Users\ASUS\Desktop\TSM_document\16Mdata1.xls"):
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
    excel1 = xlrd.open_workbook(r"C:\Users\ASUS\Desktop\TSM_document\atomic.xls")
    all_sheet1 = excel1.sheets()
    atomicdata = all_sheet1[0]
    ele1 = atomicdata.col_values(0)
    elene1 = atomicdata.col_values(1)
    elena1 = atomicdata.col_values(2)
    eleip1 = atomicdata.col_values(3)
    rcov1 = atomicdata.col_values(4)
    excel2 = xlrd.open_workbook(r"C:\Users\ASUS\Desktop\TSM_document\xenonpy.xls")
    all_sheet2 = excel2.sheets()
    xenondata = all_sheet2[0]
    polar = xenondata.col_values(57)
    fcc = xenondata.col_values(25)
    excel3 = xlrd.open_workbook(r"C:\Users\ASUS\Desktop\TSM_document\Econfig.xls")
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

        datapoint = [fccsq, max(rc), min(ea), easq, min(en), ensq, max(pl), min(pl), plsq, max(ve), min(ve), vesq, tv]
        for k in range(13-dim):
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
    return torch.tensor(Xtot, dtype=torch.float32), Ytot



from sklearn.linear_model import LogisticRegression
ytot=0.
tot=0.


X1, X0, true_y = generate_suredata(dim=13)
test_x, test_y = generate_guessdata(dim=13)
test_x=test_x.numpy()
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

    train_x = torch.cat((X0[trainset0], X1[trainset1]), 0).numpy()
    #test_x = torch.cat((X0[testset0], X1[testset1]), 0).numpy()
    y = []
    for i in range(241):
        y.append(0.)
    for i in range(56):
        y.append(1.)
    yt = []
    for i in range(60):
        yt.append(0.)
    for i in range(14):
        yt.append(1.)
    clf = LogisticRegression(random_state=0).fit(train_x, y)
    pre = clf.predict(test_x)
    for j in range(len(pre)):
        tot=tot+1.
        if pre[j]==test_y[j]:
            ytot=ytot+1.
print(ytot/tot)