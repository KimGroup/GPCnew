import math
import torch
import gpytorch
import xlwt

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
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution


def generate_data(file_loc=r"C:\Users\ROG\Desktop\test1\test.xls"):
    # X data generation
    excel = xlrd.open_workbook(file_loc)

    all_sheet = excel.sheets()

    database = all_sheet[0]
    result = database.col_values(0)[3:]
    matname = database.col_values(1)[3:]
    dsq = database.col_values(2)[3:]
    dv = database.col_values(3)[3:]
    adata = database.col_values(10)[3:]
    cdata = database.col_values(11)[3:]

    ele0 = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
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
            for k in range(len(ele0)):
                if (elements[j] == ele0[k]):
                    ip.append(eleip[k])
                    ea.append(elena[k])
                    en.append(elene[k])
                    rc.append(rcov[k])
        ipmax = max(ip)
        eamax = max(ea)
        enmax = max(en)
        ipmin = min(ip)
        eamin = min(ea)
        enmin = min(en)
        rmax = max(rc)
        rmin = min(rc)
        datapoint = [eamax, enmax, ipmax, eamin, enmin, ipmin, adata[i], cdata[i], dsq[i], dv[i], rmax, rmin,
                     1. / eamax, 1. / enmax, 1. / ipmax, 1. / eamin, 1. / enmin, 1. / ipmin, 1. / adata[i],
                     1. / cdata[i], 1. / dsq[i], 1. / dv[i], 1. / rmax, 1. / rmin]
        Xtot.append(datapoint)

    print(len(Xtot), len(Xtot[0]))
    Xraw=copy.deepcopy(Xtot)
    for i in range(len(Xtot[0])):
        max1 = 0
        min1 = 0
        for j in range(len(Xtot)):
            if Xtot[j][i] > max1:
                max1 = Xtot[j][i]
            if Xtot[j][i] < min1:
                min1 = Xtot[j][i]
        ruler = max1 - min1
        for j in range(len(Xtot)):
            Xtot[j][i] = Xtot[j][i] - min1
            Xtot[j][i] = Xtot[j][i] / ruler

    labels = [x[0] == "y" for x in result]
    labels = torch.tensor(labels) * 1
    return Xraw, torch.tensor(Xtot, dtype=torch.float32), labels


class ProjectedExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood=None, rank=1):
        if likelihood is None:
            likelihood = GaussianLikelihood()

        super(ProjectedExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module_projection = RBFKernel(
            ard_num_dims=rank,
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, 3.),
        )
        self.register_parameter(
            "projection", torch.nn.Parameter(torch.randn(train_x.shape[-1], rank, requires_grad=True))
        )
        # self.register_constraint(
        #    "projection",gpytorch.constraints.Interval(0.01, 10.)
        # )
        self.covar_module_ard = RBFKernel(
            ard_num_dims=train_x.shape[-1],
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, 3.)
        )
        self.likelihood = likelihood

    def forward(self, x):
        proj_x = x.matmul(self.projection)

        mean_x = self.mean_module(x)

        # this kernel is exp(-l_1^2 (x - x')P P^T(x - x') - l_2^2 (x - x')D(x - x'))
        # because we compute the product elementwise
        covar_x = self.covar_module_projection(proj_x) * self.covar_module_ard(x)

        return MultivariateNormal(mean_x, covar_x)


Xlist, Dptt, true_y = generate_data()
true_yf=true_y.float()


shuffled_inds = torch.randperm(Dptt.shape[0])
trainset = shuffled_inds[:1000]
testset = shuffled_inds[1000:]


class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, rank, interval1):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module_projection = RBFKernel(
            ard_num_dims=rank,
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1),
        )
        self.register_parameter(
            "projection", torch.nn.Parameter(torch.randn(train_x.shape[-1], rank, requires_grad=True))
        )
        # self.register_constraint(
        #    "projection",gpytorch.constraints.Interval(0.01, 10.)
        # )
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



likelihood = DirichletClassificationLikelihood(true_y[trainset], learn_additional_noise=True)
model = DirichletGPModel(Dptt[trainset], likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes, rank=1, interval1=3.)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
mll = ExactMarginalLogLikelihood(model.likelihood, model)

for i in range(1000):
    loss = -mll(model(Dptt[trainset]), true_y[trainset]).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (i+1) % 100 == 0:
        print("Step: ", i, "Lengthscales: ", model.covar_module_ard.lengthscale.data, "Loss: ", loss.item())

likelihood.eval()
model.eval()

with torch.no_grad():
    test_dist = model(Dptt[testset])

    pred_means = test_dist.loc

    y_pred = []

    for i in range(len(pred_means[0])):
        if pred_means[0][i] > pred_means[1][i]:
            y_pred.append(0.)
        else:
            y_pred.append(1.)
count = 0.0
testy = true_y[testset].float()
for i in range(len(testy)):
    if y_pred[i] == testy[i]:
        count = count + 1.
accuracy = count / 279.0
print(accuracy)


estimated_covar = model.projection / model.covar_module_projection.lengthscale @ model.projection.t() + \
    torch.diag(model.covar_module_ard.lengthscale.reciprocal())

from matplotlib import pyplot as plt
plt.imshow(estimated_covar.data)
plt.colorbar()
plt.show()

covar_inv_diags = estimated_covar.diag() ** 0.5
estimated_corr = estimated_covar / torch.outer(covar_inv_diags, covar_inv_diags)
ref=[['eamax','enmax','ipmax','eamin','enmin','ipmin','adata','cdata','dsq','dv','rmax','rmin','1./eamax','1./enmax','1./ipmax','1./eamin','1./enmin','1./ipmin','1./adata','1./cdata','1./dsq','1./dv','1./rmax','1./rmin']]
cormatrix=estimated_corr.data.tolist()
plt.imshow(estimated_corr.data)
plt.colorbar()
plt.show()
#print(estimated_corr.data.numpy())
ttt=1.-torch.abs(estimated_corr.data)
ttt=ttt.numpy()*50.0
#print(ttt)
for i in range(len(ttt[0])):
    ttt[i][i]=0.0
    for j in range(i+1, len(ttt[0])):
        ttt[j][i]=ttt[i][j]
from scipy.spatial.distance import squareform

condensed_dist = squareform(ttt)
cc=[]
for i in range(len(ttt[0])):
    for j in range(i+1,len(ttt[0])):
        cc.append((ttt[i][j]))

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt


def hierarchy_cluster(data, method='average', threshold=2.5):
    '''层次聚类

    Arguments:
        data [[0, float, ...], [float, 0, ...]] -- 文档 i 和文档 j 的距离

    Keyword Arguments:
        method {str} -- [linkage的方式： single、complete、average、centroid、median、ward] (default: {'average'})
        threshold {float} -- 聚类簇之间的距离
    Return:
        cluster_number int -- 聚类个数
        cluster [[idx1, idx2,..], [idx3]] -- 每一类下的索引
    '''
    data = np.array(data)

    Z = linkage(data, method=method)
    cluster_assignments = fcluster(Z, threshold, criterion='distance')
    print(type(cluster_assignments))
    num_clusters = cluster_assignments.max()
    indices = get_cluster_indices(cluster_assignments)

    return num_clusters, indices


def get_cluster_indices(cluster_assignments):
    '''映射每一类至原数据索引

    Arguments:
        cluster_assignments 层次聚类后的结果

    Returns:
        [[idx1, idx2,..], [idx3]] -- 每一类下的索引
    '''
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])

    return indices


num_clusters, indices = hierarchy_cluster(condensed_dist)

print("%d clusters" % num_clusters)
for k, ind in enumerate(indices):
    print("cluster", k + 1, "is", ind)



raw=estimated_covar.data.tolist()
onechain=[]
ordchain=[]
for i in range(len(cormatrix)):
    onechain.append(cormatrix[i][i])
    ordchain.append(i)
for i in range(len(onechain)):
    maxx=onechain[i]
    anchor=i
    for j in range(i+1, len(onechain)):
        if onechain[j]>maxx:
            maxx=onechain[j]
            anchor=j
    if maxx!=onechain[i]:
        onechain[anchor]=onechain[i]
        onechain[i]=maxx
        place=ordchain[i]
        ordchain[i]=ordchain[anchor]
        ordchain[anchor]=place
for i in range(24):
    print(ordchain[i])
average=[]
clusters=[]
for k, ind in enumerate(indices):
    summ=0
    clusters.append(ind)
    if len(ind)==1:
        average.append(0)
    else:
        for i in range(len(ind)):
            for j in range(i+1,len(ind)):
                summ=summ+abs(cormatrix[ind[i]][ind[j]])
        summ=summ*2.0/(float(len(ind))*float(len(ind)-1))
        average.append(summ)
print(average)

for i in range(len(average)):
    maxx=average[i]
    for j in range(i+1, len(average)):
        if average[j]>maxx:
            average[i]=average[j]
            average[j]=maxx
            maxx=average[i]
            buffer=clusters[i]
            clusters[i]=clusters[j]
            clusters[j]=buffer
for i in range(len(clusters)):
    if len(clusters[i])>3:
        cluster=clusters[i]
        break
print(cluster)

pairs=[]
strength=[]
for i in range(len(cluster)-1):
    for j in range(i+1, len(cluster)):
        pairs.append([cluster[i], cluster[j]])
        strength.append(abs(cormatrix[cluster[i]][cluster[j]]))

for i in range(len(strength)-1):
    maxx=strength[i]
    for j in range(i+1, len(strength)):
        if strength[j]>maxx:
            strength[i]=strength[j]
            strength[j]=maxx
            maxx=strength[i]
            buffer=pairs[i]
            pairs[i]=pairs[j]
            pairs[j]=buffer

ref.append([])
xbuffer=[]
for i in range(len(Xlist)):
    xbuffer.append([])
count=10
if len(pairs)<count:
    count=len(pairs)
for i in range(count):
    ref[1].append('('+ref[0][pairs[i][0]]+'+'+ref[0][pairs[i][1]]+')')
    ref[1].append('|'+ref[0][pairs[i][0]] + '-' + ref[0][pairs[i][1]]+'|')
    ref[1].append('(' + ref[0][pairs[i][0]] + '*' + ref[0][pairs[i][1]] + ')')
    ref[1].append('1'+r'/'+'(' + ref[0][pairs[i][0]] + '*' + ref[0][pairs[i][1]] + ')')
    for j in range(len(Xlist)):
        xbuffer[j].append(Xlist[j][pairs[i][0]]+Xlist[j][pairs[i][1]])
        xbuffer[j].append(abs(Xlist[j][pairs[i][0]] - Xlist[j][pairs[i][1]]))
        xbuffer[j].append(Xlist[j][pairs[i][0]]*Xlist[j][pairs[i][1]])
        xbuffer[j].append(1./(Xlist[j][pairs[i][0]]*Xlist[j][pairs[i][1]]))

for i in range(len(xbuffer[0])):
    max1 = 0
    min1 = 0
    for j in range(len(xbuffer)):
        if xbuffer[j][i] > max1:
            max1 = xbuffer[j][i]
        if xbuffer[j][i] < min1:
            min1 = xbuffer[j][i]
    ruler = max1 - min1
    for j in range(len(xbuffer)):
        xbuffer[j][i] = xbuffer[j][i] - min1
        xbuffer[j][i] = xbuffer[j][i] / ruler
print(ref[1])
Dptt=torch.tensor(xbuffer, dtype=torch.float32)











likelihood = DirichletClassificationLikelihood(true_y[trainset], learn_additional_noise=True)
model = DirichletGPModel(Dptt[trainset], likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes, rank=1, interval1=5.)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.2)
mll = ExactMarginalLogLikelihood(model.likelihood, model)

for i in range(2000):
    loss = -mll(model(Dptt[trainset]), true_y[trainset]).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (i+1) % 100 == 0:
        print("Step: ", i, "Lengthscales: ", model.covar_module_ard.lengthscale.data, "Loss: ", loss.item())

likelihood.eval()
model.eval()

with torch.no_grad():
    test_dist = model(Dptt[testset])

    pred_means = test_dist.loc

    y_pred = []

    for i in range(len(pred_means[0])):
        if pred_means[0][i] > pred_means[1][i]:
            y_pred.append(0.)
        else:
            y_pred.append(1.)
count = 0.0
testy = true_y[testset].float()
for i in range(len(testy)):
    if y_pred[i] == testy[i]:
        count = count + 1.
accuracy = count / 279.0
print(accuracy)


estimated_covar = model.projection / model.covar_module_projection.lengthscale @ model.projection.t() + \
    torch.diag(model.covar_module_ard.lengthscale.reciprocal())

print('here')
plt.imshow(estimated_covar.data)
plt.colorbar()
plt.show()
print('here')
covar_inv_diags = estimated_covar.diag() ** 0.5
estimated_corr = estimated_covar / torch.outer(covar_inv_diags, covar_inv_diags)

cormatrix=estimated_corr.data.tolist()
plt.imshow(cormatrix)
plt.colorbar()
plt.show()







plt.imshow(wet)
plt.colorbar()
plt.savefig('correlation_fig.png')