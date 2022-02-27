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




def generate_data(dim, file_loc=r"16Mdata.xls"):
    # X data generation
    excel = xlrd.open_workbook(file_loc)

    all_sheet = excel.sheets()

    database = all_sheet[0]
    result = database.col_values(0)[3:]
    matname = database.col_values(1)[3:]
    dsq = database.col_values(2)[3:]
    dv = database.col_values(3)[3:]
    sqatom = database.col_values(8)[3:]
    adata = database.col_values(10)[3:]
    cdata = database.col_values(11)[3:]
    # add
    excel1 = xlrd.open_workbook(r"atomic.xls")
    all_sheet1 = excel1.sheets()
    atomicdata = all_sheet1[0]
    ele1 = atomicdata.col_values(0)
    elene1 = atomicdata.col_values(1)
    elena1 = atomicdata.col_values(2)
    eleip1 = atomicdata.col_values(3)
    rcov1 = atomicdata.col_values(4)
    print(elene1)
    print(elena1)
    """ele0 = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
            'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
            'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
            'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
            'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu',
            'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
            'No', 'Lr']
    elene = [2.20, 0, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0, 0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0, 0.82,
             1.00, 1.36, 1.54, 1.63, 1.66,
             1.55, 1.83, 1.88, 1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 2.55, 2.96, 3.00, 0.82, 0.95, 1.22, 1.33, 1.6, 2.16,
             1.9, 2.2, 2.28, 2.2, 1.93, 1.69, 1.78,
             1.96, 2.05, 2.1, 2.66, 2.60, 0.79, 0.89, 1.1, 1.12, 1.13, 1.14, 1.13, 1.17, 1.2, 1.2, 1.1, 1.22, 1.23,
             1.24, 1.25, 1.1, 1.27, 1.3, 1.5, 2.36, 1.9,
             2.2, 2.20, 2.28, 2.54, 2.0, 1.62, 1.87, 2.02, 2.0, 2.2, 2.2, 0.795, 0.9, 1.1, 1.3, 1.5, 1.38, 1.36, 1.28,
             1.13, 1.28, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3]
    elena = [72.769, -48.2, 59.633, -48.2, 26.989, 121.776, -6.8, 140.976, 328.165, -116.19, 52.867, -40.19, 41.763,
             134.068, 72.037, 200.410, 348.575,
             -96.2, 48.383, 2.37, 18.2, 7.289, 50.911, 65.21, -50.2, 14.785, 63.898, 111.65, 119.235, -58.2, 29.061,
             118.935, 77.65, 194.959, 324.537, -96.2,
             46.884, 5.023, 29.6, 41.807, 88.517, 72.1, 53.2, 100.96, 110.27, 54.24, 125.862, -68.2, 37.043, 107.298,
             101.059, 190.161, 295.153, -77.2, 45.505,
             13.954, 53.795, 55.2, 10.539, 9.406, 12.45, 15.63, 11.2, 13.22, 12.670, 33.96, 32.61, 30.10, 99.3, -1.93,
             23.04, 17.18, 31.2, 78.76, 5.8273, 103.99,
             150.94, 205.041, 222.747, -48.2, 36.414, 34.4183, 90.924, 136.7, 223.12, -68.2, 46.89, 9.6485, 33.77,
             112.72, 53.03, 50.94, 45.85, -48.33, 9.93,
             27.17, -165.24, -97.31, -28.6, 33.96, 93.91, -223.22, -30.04]
    rcov = [31.5, 28, 128.7, 96.3, 84.3, 76.1, 71.1, 66.2, 57.3, 58, 166.9, 141.7, 121.4, 111.2, 107.3, 105.3, 102.4,
            106.1, 203.1, 176.1, 170.7, 160.8, 153.8,
            139.5, 139.5, 132.3, 126.3, 124.4, 132.4, 122.4, 122.3, 120.4, 119.4, 120.4, 120.3, 116.4, 220.9, 195.1,
            190.7, 175.7, 164.6, 154.5, 147.7, 146.7,
            142.7, 139.6, 145.5, 144.9, 142.5, 139.4, 139.5, 138.4, 139.3, 140.9, 244.1, 215.1, 207.8, 204.9, 203.7,
            201.6, 199, 198.8, 198.6, 196.6, 194.5,
            192.7, 192.7, 189.6, 190.1, 187.8, 175.1, 187.8, 170.8, 162.7, 151.7, 144.4, 141.6, 136.5, 136.6, 132.5,
            145.7, 146.5, 148.4, 140.4, 150, 150, 260,
            221.2, 215, 206.6, 200, 196.7, 190.1, 187.1, 180.6, 169.3, 166, 168, 165, 167, 173, 176, 161]
    eleipabs = [13.60, 24.59, 5.392, 9.323, 8.298, 11.26, 14.53, 13.62, 17.42, 21.56, 5.139, 7.646, 5.986, 8.152, 10.49,
                10.36, 12.97, 15.76, 4.341, 6.113, 6.562,
                6.828, 6.746, 6.767, 7.434, 7.902, 7.881, 7.640, 7.726, 9.394, 5.999, 7.899, 9.789, 9.752, 11.81, 14.00,
                4.177, 5.695, 6.217, 6.634, 6.759, 7.092,
                7.280, 7.361, 7.459, 8.337, 7.576, 8.994, 5.786, 7.344, 8.608, 9.010, 10.45, 12.13, 3.894, 5.212, 5.577,
                5.539, 5.473, 5.525, 5.582, 5.644, 5.670,
                6.150, 5.864, 5.939, 6.022, 6.108, 6.184, 6.254, 5.426, 6.825, 7.550, 7.864, 7.834, 8.438, 8.967, 8.959,
                9.226, 10.44, 6.108, 7.417, 7.286, 8.417,
                9.318, 10.75, 4.073, 5.278, 5.17, 6.307, 5.89, 6.194, 6.266, 6.026, 5.974, 5.992, 6.198, 6.282, 6.42,
                6.50, 6.58, 6.65, 4.9]


    eleip = []
    for i in eleipabs:
        eleip.append(-i)
    """
    Xtot = []
    Ytot = []
    labels = [x[0] == "y" for x in result]
    for i in range(len(matname)):
        elements = []
        for j in range(len(matname[i])):
            if (ord(matname[i][j]) > 64) & (ord(matname[i][j]) < 91):
                ele = matname[i][j]
                if j < (len(matname[i]) - 1):
                    if (ord(matname[i][j + 1]) > 96) & (ord(matname[i][j + 1]) < 123):
                        ele = ele + matname[i][j + 1]
                rep = 0
                for k in elements:
                    if (j == ele):
                        rep = 1
                if (rep == 0):
                    elements.append(ele)
        ea, ip, en, rc = [], [], [], []

        for j in range(len(elements)):
            for k in range(len(ele1)):
                if (elements[j] == ele1[k]):
                    ip.append(eleip1[k])
                    ea.append(elena1[k])
                    en.append(elene1[k])
                    rc.append(rcov1[k])
        ipmax = max(ip)
        eamax = max(ea)
        enmax = max(en)
        ipmin = min(ip)
        eamin = min(ea)
        enmin = min(en)
        rmax = max(rc)
        rmin = min(rc)
        for k in range(len(ele1)):
            if sqatom[i] == ele1[k]:
                ipsq = eleip1[k]
                easq = elena1[k]
                ensq = elene1[k]
                rcsq = rcov1[k]
        datapoint =[rmax, enmin, eamin, eamax, rmin, dsq[i], dv[i], easq, cdata[i], ipsq]
            #[rmax, enmin, eamin, eamax, rmin, dsq[i], dv[i], easq, cdata[i]]

            #[rmax, enmin, eamin, eamax, rmin, dsq[i], dv[i], easq, cdata[i]]
            #[cdata[i],dsq[i],rmax,enmin,eamin,dv[i],rmin,easq,eamax]
            #[eamin, easq, cdata[i], dv[i], eamax, dsq[i], rmax, rmin, enmin]
        for ii in range(10-dim):
            del(datapoint[0])
        #[eamax, ipmax, eamin, enmin, adata[i], scaledcdata[i], dsq[i], dv[i], rmax,easq,enmax, rmin]
        #[eamax, enmax, ipmax, eamin, enmin, adata[i], scaledcdata[i], dsq[i], dv[i], rmax, rmin , easq]
        if (dsq[i]>2.25)and(dsq[i]<3.75)and(dv[i]<3.5)and(cdata[i]<15.)and(rmax<220)and(rmin>40):
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
    print(len(Xtot), len(Ytot))
    return Xraw, torch.tensor(Xtot, dtype=torch.float32), Ytot



def vary_data(dim, unfixed1, unfixed2, file_loc=r"16Mdata.xls"):
    # X data generation
    excel = xlrd.open_workbook(file_loc)

    all_sheet = excel.sheets()

    database = all_sheet[0]
    result = database.col_values(0)[3:]
    matname = database.col_values(1)[3:]
    dsq = database.col_values(2)[3:]
    dv = database.col_values(3)[3:]
    sqatom = database.col_values(8)[3:]
    adata = database.col_values(10)[3:]
    cdata = database.col_values(11)[3:]
    # add
    excel1 = xlrd.open_workbook(r"atomic.xls")
    all_sheet1 = excel1.sheets()
    atomicdata = all_sheet1[0]
    ele1 = atomicdata.col_values(0)
    elene1 = atomicdata.col_values(1)
    elena1 = atomicdata.col_values(2)
    eleip1 = atomicdata.col_values(3)
    rcov1 = atomicdata.col_values(4)
    print(elene1)
    print(elena1)
    """ele0 = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
            'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
            'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
            'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
            'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu',
            'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
            'No', 'Lr']
    elene = [2.20, 0, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0, 0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0, 0.82,
             1.00, 1.36, 1.54, 1.63, 1.66,
             1.55, 1.83, 1.88, 1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 2.55, 2.96, 3.00, 0.82, 0.95, 1.22, 1.33, 1.6, 2.16,
             1.9, 2.2, 2.28, 2.2, 1.93, 1.69, 1.78,
             1.96, 2.05, 2.1, 2.66, 2.60, 0.79, 0.89, 1.1, 1.12, 1.13, 1.14, 1.13, 1.17, 1.2, 1.2, 1.1, 1.22, 1.23,
             1.24, 1.25, 1.1, 1.27, 1.3, 1.5, 2.36, 1.9,
             2.2, 2.20, 2.28, 2.54, 2.0, 1.62, 1.87, 2.02, 2.0, 2.2, 2.2, 0.795, 0.9, 1.1, 1.3, 1.5, 1.38, 1.36, 1.28,
             1.13, 1.28, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3]
    elena = [72.769, -48.2, 59.633, -48.2, 26.989, 121.776, -6.8, 140.976, 328.165, -116.19, 52.867, -40.19, 41.763,
             134.068, 72.037, 200.410, 348.575,
             -96.2, 48.383, 2.37, 18.2, 7.289, 50.911, 65.21, -50.2, 14.785, 63.898, 111.65, 119.235, -58.2, 29.061,
             118.935, 77.65, 194.959, 324.537, -96.2,
             46.884, 5.023, 29.6, 41.807, 88.517, 72.1, 53.2, 100.96, 110.27, 54.24, 125.862, -68.2, 37.043, 107.298,
             101.059, 190.161, 295.153, -77.2, 45.505,
             13.954, 53.795, 55.2, 10.539, 9.406, 12.45, 15.63, 11.2, 13.22, 12.670, 33.96, 32.61, 30.10, 99.3, -1.93,
             23.04, 17.18, 31.2, 78.76, 5.8273, 103.99,
             150.94, 205.041, 222.747, -48.2, 36.414, 34.4183, 90.924, 136.7, 223.12, -68.2, 46.89, 9.6485, 33.77,
             112.72, 53.03, 50.94, 45.85, -48.33, 9.93,
             27.17, -165.24, -97.31, -28.6, 33.96, 93.91, -223.22, -30.04]
    rcov = [31.5, 28, 128.7, 96.3, 84.3, 76.1, 71.1, 66.2, 57.3, 58, 166.9, 141.7, 121.4, 111.2, 107.3, 105.3, 102.4,
            106.1, 203.1, 176.1, 170.7, 160.8, 153.8,
            139.5, 139.5, 132.3, 126.3, 124.4, 132.4, 122.4, 122.3, 120.4, 119.4, 120.4, 120.3, 116.4, 220.9, 195.1,
            190.7, 175.7, 164.6, 154.5, 147.7, 146.7,
            142.7, 139.6, 145.5, 144.9, 142.5, 139.4, 139.5, 138.4, 139.3, 140.9, 244.1, 215.1, 207.8, 204.9, 203.7,
            201.6, 199, 198.8, 198.6, 196.6, 194.5,
            192.7, 192.7, 189.6, 190.1, 187.8, 175.1, 187.8, 170.8, 162.7, 151.7, 144.4, 141.6, 136.5, 136.6, 132.5,
            145.7, 146.5, 148.4, 140.4, 150, 150, 260,
            221.2, 215, 206.6, 200, 196.7, 190.1, 187.1, 180.6, 169.3, 166, 168, 165, 167, 173, 176, 161]
    eleipabs = [13.60, 24.59, 5.392, 9.323, 8.298, 11.26, 14.53, 13.62, 17.42, 21.56, 5.139, 7.646, 5.986, 8.152, 10.49,
                10.36, 12.97, 15.76, 4.341, 6.113, 6.562,
                6.828, 6.746, 6.767, 7.434, 7.902, 7.881, 7.640, 7.726, 9.394, 5.999, 7.899, 9.789, 9.752, 11.81, 14.00,
                4.177, 5.695, 6.217, 6.634, 6.759, 7.092,
                7.280, 7.361, 7.459, 8.337, 7.576, 8.994, 5.786, 7.344, 8.608, 9.010, 10.45, 12.13, 3.894, 5.212, 5.577,
                5.539, 5.473, 5.525, 5.582, 5.644, 5.670,
                6.150, 5.864, 5.939, 6.022, 6.108, 6.184, 6.254, 5.426, 6.825, 7.550, 7.864, 7.834, 8.438, 8.967, 8.959,
                9.226, 10.44, 6.108, 7.417, 7.286, 8.417,
                9.318, 10.75, 4.073, 5.278, 5.17, 6.307, 5.89, 6.194, 6.266, 6.026, 5.974, 5.992, 6.198, 6.282, 6.42,
                6.50, 6.58, 6.65, 4.9]


    eleip = []
    for i in eleipabs:
        eleip.append(-i)
    """
    Xtot = []
    for i in range(len(matname)):
        elements = []
        for j in range(len(matname[i])):
            if (ord(matname[i][j]) > 64) & (ord(matname[i][j]) < 91):
                ele = matname[i][j]
                if j < (len(matname[i]) - 1):
                    if (ord(matname[i][j + 1]) > 96) & (ord(matname[i][j + 1]) < 123):
                        ele = ele + matname[i][j + 1]
                rep = 0
                for k in elements:
                    if (j == ele):
                        rep = 1
                if (rep == 0):
                    elements.append(ele)
        ea, ip, en, rc = [], [], [], []

        for j in range(len(elements)):
            for k in range(len(ele1)):
                if (elements[j] == ele1[k]):
                    ip.append(eleip1[k])
                    ea.append(elena1[k])
                    en.append(elene1[k])
                    rc.append(rcov1[k])
        ipmax = max(ip)
        eamax = max(ea)
        enmax = max(en)
        ipmin = min(ip)
        eamin = min(ea)
        enmin = min(en)
        rmax = max(rc)
        rmin = min(rc)
        for k in range(len(ele1)):
            if sqatom[i] == ele1[k]:
                ipsq = eleip1[k]
                easq = elena1[k]
                ensq = elene1[k]
                rcsq = rcov1[k]

        datapoint = [rmax, enmin, eamin, eamax, rmin, dsq[i], dv[i], easq, cdata[i], ipsq]
            #[rmax, enmin, eamin, eamax, rmin, dsq[i], dv[i], easq, cdata[i]]

            #[rmax, enmin, eamin, eamax, rmin, dsq[i], dv[i], easq, cdata[i]]
            #[cdata[i],dsq[i],rmax,enmin,eamin,dv[i],rmin,easq,eamax]
            #[eamin, easq, cdata[i], dv[i], eamax, dsq[i], rmax, rmin, enmin]
        for i in range(10-dim):
            del(datapoint[0])
        #[eamax, ipmax, eamin, enmin, adata[i], scaledcdata[i], dsq[i], dv[i], rmax,easq,enmax, rmin]
        #[eamax, enmax, ipmax, eamin, enmin, adata[i], scaledcdata[i], dsq[i], dv[i], rmax, rmin , easq]
        if (dsq[i] > 2.25) and (dsq[i] < 3.75) and (dv[i] < 3.5) and (cdata[i] < 15.) and (rmax < 220) and (rmin > 40):
            Xtot.append(datapoint)  # eamin, enmin, ipmin,    , rmin

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

    for i in range(len(Xtot[0])):
        if (i!=unfixed1)and(i!=unfixed2):
            sum1=0
            for j in range(len(Xtot)):
                sum1+=Xtot[j][i]
            sum1=sum1/float(len(Xtot))
            for j in range(len(Xtot)):
                Xtot[j][i]=sum1


    return Xtot[0]

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
    from botorch.optim.fit import fit_gpytorch_torch

    likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
    model = DirichletGPModel(
        train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes, initialization=inbuffer,
        rank=rank, interval1=interval1
    )
    model = model.to(train_x)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _, info_dict = fit_gpytorch_torch(mll, options={"maxiter": 1000, "lr": lr})
    print("Final MLL: ", info_dict["fopt"])

    return model, info_dict["fopt"]


def compute_accuracy(model, likelihood, test_x, test_y, train_x, train_y):
    likelihood.eval()
    model.eval()

    with torch.no_grad():
        test_dist = model(test_x)

        pred_means = test_dist.loc

        y_pred = []

        for i in range(len(pred_means[0])):
            if pred_means[0][i] > pred_means[1][i]:
                y_pred.append(0.)
            else:
                y_pred.append(1.)
    count = 0.0
    testy = test_y.float()
    for i in range(len(testy)):
        if y_pred[i] == testy[i]:
            count = count + 1.
    accuracy = count / test_x.shape[0]
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
    return accuracy, accuracy2


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



basis=torch.randn(10, 3, dtype=torch.float32)/3.
for d in range(10,9,-1):
    Xlist, Dptt, true_y = generate_data(dim=d)
    Dptt = Dptt.to(device, dtype)
    true_y = true_y.to(device)
    true_yf = true_y.float()
    inbuffer=[]
    for i in range(d):
        inbuffer.append([])
        for j in range(len(basis[i+10-d])):
            inbuffer[i].append(basis[i+10-d][j])
    for u1 in range(0,2):
        for u2 in range(u1+1,10):
            aaa = []
            for i in range(21):
                aaa.append([])
                for j in range(21):
                    aaa[i].append(0.)
            posterior1 = torch.tensor(aaa, dtype=torch.float32)
            posterior0 = torch.tensor(aaa, dtype=torch.float32)
            prob0 = torch.tensor(aaa, dtype=torch.float32)
            prob1 = torch.tensor(aaa, dtype=torch.float32)

            for i in range(30):
                print("trying seed: ", i)
                torch.random.manual_seed(i)

                # note that now the training set is not fixed
                # lets use 80%
                shuffled_inds = torch.randperm(Dptt.shape[0])
                trainset = shuffled_inds[:1023]
                testset = shuffled_inds[1023:]

                with gpytorch.settings.max_cholesky_size(2000):
                    model, mll1 = make_and_fit_classifier(Dptt[trainset], true_y[trainset], inbuffer=inbuffer,
                                                          lr=0.1)
                    # acc, acc2 = compute_accuracy(model, model.likelihood, Dptt[testset], true_y[testset],
                    # Dptt[trainset], true_y[trainset])
                    # rmatrix, smatrix, cvector1 = cluster_lengthscales(model)
                # summatrix = summatrix + smatrix
                # rawmatrix = rawmatrix + rmatrix
                # cvectors = cvectors + cvector1
                # r1=r1+r11
                # acc_list.append(acc)
                # acc_list2.append(acc2)
                # mll_list.append(mll1)
                model.likelihood.eval()
                model.eval()
                unfix1 = u1
                unfix2 = u2
                fixed = vary_data(dim=10, unfixed1=unfix1, unfixed2=unfix2)
                d1 = []
                d2 = []
                for j in range(21):
                    d1.append([])
                    d2.append([])
                    value1 = float(j) / 20.
                    for k in range(21):
                        d1[j].append(value1)
                    fixed[unfix1] = value1
                    for k in range(21):
                        value2 = float(k) / 20.
                        d2[j].append(value2)
                        fixed[unfix2] = value2

                        point = torch.tensor([fixed], dtype=torch.float32)
                        point = point.to(device)
                        with torch.no_grad():
                            test_dist = model(point)
                            pred_samples = test_dist.sample(torch.Size((256,))).exp()
                            stdprobabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).std(0)
                            pred_means = test_dist.loc
                            pred_means = pred_means.cpu()
                            prob0[j][k] = prob0[j][k] + stdprobabilities[0]
                            prob1[j][k] = prob1[j][k] + stdprobabilities[1]
                            posterior0[j][k] = posterior0[j][k] + np.exp(pred_means[0][0])
                            posterior1[j][k] = posterior1[j][k] + np.exp(pred_means[1][0])

            post0 = posterior0.data.div(30)
            post1 = posterior1.data.div(30)
            prob0 = prob0.data.div(30)
            prob1 = prob1.data.div(30)
            levels = np.linspace(0, 0.4, 50)
            plt.figure()
            im = plt.contourf(d1, d2, prob0, levels=levels)
            plt.colorbar(im)
            plt.show()
            plt.savefig(str(u1)+str(u2)+'stdProb0.png')
            plt.figure()
            im = plt.contourf(d1, d2, prob1, levels=levels)
            plt.colorbar(im)
            plt.show()
            plt.savefig(str(u1)+str(u2)+'stdProb1.png')
            """plt.figure()
            im = plt.contourf(d1, d2, post0, levels=levels)
            plt.colorbar(im)
            plt.show()
            plt.savefig(str(u1)+str(u2)+'Posterior0.png')
            plt.figure()
            im = plt.contourf(d1, d2, prob1, levels=levels)
            plt.colorbar(im)
            plt.show()
            plt.savefig(str(u1)+str(u2)+'Posterior1.png')"""


