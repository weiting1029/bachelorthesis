# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from numpy.random import multivariate_normal
from numpy.random import normal
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV
from scipy import stats
from matplotlib import pyplot
import warnings
import random
import func
import time
import openpyxl

warnings.filterwarnings("ignore")
# warnings.simplefilter("default")
from numpy import linalg as LA
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# simulation setting
# n = 100
# p = 50
# Sigma1 = np.ones([p,p])

# def main():
#     n = 100
#     p = 50
#
#     Sigma1 = 0.8 * np.ones([p, p])
#     Sigma2 = np.zeros([p, p])
#     rho = 0.6
#     for i in range(p):
#         for j in range(p):
#             Sigma2[i, j] = rho ** abs(i - j)
#
#     np.fill_diagonal(Sigma1, 1)
#     mean1 = np.zeros(p)
#     random.seed(10)
#     Q = multivariate_normal(mean=mean1, cov=Sigma2, size=n)
#
#     mu, sigma = 0, 0.1
#     ERR = normal(mu, sigma, n)
#
#     # set the true parameter
#     beta = np.zeros(p)
#     beta[1:3] = 1
#     beta[0] = 0
#
#     # DGP1
#     # Y = ERR + Q @ beta
#     seed = 10
#     KK = 150
#     alpha = 0.05
#     theta_candidate = np.linspace(0, 0.05, 20)
#
#     test_power_desf = func.test_power_simulation_desf(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma2, n)
#     test_power_ldpe = func.test_power_simulation_ldpe(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma2, n)
#
#     plt.plot(theta_candidate, test_power_desf / KK)
#     plt.plot(theta_candidate, test_power_ldpe / KK)
#     plt.ylabel('Testing Power')
#     plt.xlabel('theta')


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
sns.set_style('darkgrid')
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('font', size=13)
# main()


n = 100
p = 150

Sigma1 = 0.8 * np.ones([p, p])
Sigma2 = np.zeros([p, p])
rho = 0.6
for i in range(p):
    for j in range(p):
        Sigma2[i, j] = rho ** abs(i - j)

np.fill_diagonal(Sigma1, 1)
mean1 = np.zeros(p)
random.seed(10)
Q = multivariate_normal(mean=mean1, cov=Sigma2, size=n)

mu, sigma = 0, 0.1
ERR = normal(mu, sigma, n)

# set the true parameter
beta = np.zeros(p)
beta[1:3] = 1
beta[0] = 0

# DGP1
# Y = ERR + Q @ beta
seed = 10
KK = 250
alpha = 0.05
theta_candidate = np.linspace(0, 0.05, 20)

start = time.time()
test_power_desf = func.test_power_simulation_ldpe_graphical(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma2,
                                                            n)
end = time.time()
print(end - start)
df = pd.DataFrame(test_power_desf, columns=['DeSF'])
df.to_excel("test_power_p150_gldpe.xlsx")
# test_power_ldpe = func.test_power_simulation_ldpe(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma2, n)


sns.set_style('darkgrid')
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('font', size=13)

plt.plot(theta_candidate, test_power_desf, label='DeSF')
# plt.plot(theta_candidate, test_power_ldpe, label='LDPE')
plt.plot(theta_candidate, alpha * np.ones(len(theta_candidate)), 'r--', label='alpha = 0.05')
plt.ylabel('Test Power (Rejection Rates)')
plt.xlabel('theta')
plt.legend()
plt.title(' n  = 100, p = 50 ')
plt.savefig('test_power_p50')
plt.show()
