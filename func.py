import numpy as np
from numpy.random import multivariate_normal
from numpy.random import normal
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import LassoLarsCV
from scipy import stats
from matplotlib import pyplot
import warnings
import random

warnings.filterwarnings("ignore")
# warnings.simplefilter("default")
from numpy import linalg as LA
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def deco_score_function(est_beta, Y, Q, sigma):
    n, p = Q.shape
    X = Q[:, 1:]
    Z = Q[:, 0]
    #     reg_desf = LassoCV(cv=5, random_state=0).fit(Q, Y)
    #     est_beta = reg_desf.coef_
    #     lambda_desf = reg_desf.alpha_
    est_theta = est_beta[0]
    est_gamma = est_beta[1:]
    reg_w = LassoCV(cv=5, random_state=0).fit(X, Z)
    est_w = reg_w.coef_

    desf = -(1 / ((sigma ** 2) * n)) * np.dot(Y - X @ est_gamma, Z - X @ est_w)
    return desf


def u_score_function(est_beta, Y, Q, sigma):
    n, p = Q.shape
    X = Q[:, 1:]
    Z = Q[:, 0]
    #     reg_desf = LassoCV(cv=5, random_state=0).fit(Q, Y)
    #     est_beta = reg_desf.coef_
    #     lambda_desf = reg_desf.alpha_

    est_theta = est_beta[0]
    est_gamma = est_beta[1:]
    reg_w = LassoCV(cv=5, random_state=0).fit(X, Z)
    est_w = reg_w.coef_

    desf = -(1 / ((sigma ** 2) * n)) * np.dot(Y - X @ est_gamma, Z - X @ est_w)
    Z_matrix = np.repeat(np.reshape(Z, [n, 1]), p - 1, axis=1)  # ?
    I = 1 / ((sigma ** 2) * n) * (np.sum(Z ** 2) - np.sum(np.multiply(X, Z_matrix), axis=0) @ est_w)
    u_score = n ** (0.5) * desf * I ** (-0.5)
    return u_score


def test_power_simulation_desf(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma, n):
    #     beta[0]=0
    rej_list = np.zeros(len(theta_candidate))
    lambda_list = np.zeros(len(theta_candidate))
    for i in range(50):
        np.random.seed(seed + i)
        ERR_temp = normal(mu, sigma, n)
        #         print(random.random())
        Q_temp = multivariate_normal(mean=mean1, cov=Sigma, size=n)
        p = len(mean1)
        hyp_beta = np.zeros(p)
        hyp_beta[1:3] = 1
        for j in range(len(theta_candidate)):
            hyp_beta[0] = theta_candidate[j]
            Y_temp = ERR_temp + Q_temp @ hyp_beta
            reg_temp = LassoCV(cv=5, random_state=0).fit(Q_temp, Y_temp)
            temp_beta = reg_temp.coef_
            lambda_list[j] = lambda_list[j] + reg_temp.alpha_
            #             print(temp_beta[0])
            u_score = u_score_function(temp_beta, Y_temp, Q_temp, sigma)
            p_value = 1 - 2 * (1 - stats.norm.cdf(abs(u_score)))
            if p_value > 1 - alpha:
                rej_list[j] += 1
    lambda_list = lambda_list / 50

    for i in range(50, KK):
        np.random.seed(seed + i)
        ERR_temp = normal(mu, sigma, n)
        #         print(random.random())
        Q_temp = multivariate_normal(mean=mean1, cov=Sigma, size=n)
        p = len(mean1)
        hyp_beta = np.zeros(p)
        hyp_beta[1:3] = 1
        for j in range(len(theta_candidate)):
            hyp_beta[0] = theta_candidate[j]
            Y_temp = ERR_temp + Q_temp @ hyp_beta
            reg_temp = Lasso(alpha=lambda_list[j]).fit(Q_temp, Y_temp)
            temp_beta = reg_temp.coef_
            # lambda_list[j] = lambda_list[j] + reg_temp.alpha_
            #             print(temp_beta[0])
            u_score = u_score_function(temp_beta, Y_temp, Q_temp, sigma)
            p_value = 1 - 2 * (1 - stats.norm.cdf(abs(u_score)))
            if p_value > 1 - alpha:
                rej_list[j] += 1

    return rej_list / KK


def estimate_Sigma_inverse(Q):
    n, p = Q.shape
    est_C = np.zeros([p, p])
    score_list = np.zeros(p)
    for i in range(p):
        y = Q[:, i]
        X = np.delete(Q, i, axis=1)
        reg_i = LassoCV(cv=5, random_state=0).fit(X, y)
        gamma_i = reg_i.coef_
        alpha_i = reg_i.alpha_
        est_C[i] = np.insert(-gamma_i, i, 1)
        score_list[i] = LA.norm(y - X @ gamma_i, 2) ** 2 / n + alpha_i * LA.norm(gamma_i, 1)
    est_T = np.diag(score_list ** (-2))
    return est_T @ est_C


def LDPE_beta(est_beta, Y, Q):
    n, p = Q.shape
    est_C = np.zeros([p, p])
    score_list = np.zeros(p)
    for i in range(p):
        y = Q[:, i]
        X = np.delete(Q, i, axis=1)
        reg_i = LassoCV(cv=5, random_state=0).fit(X, y)
        gamma_i = reg_i.coef_
        alpha_i = reg_i.alpha_
        est_C[i] = np.insert(-gamma_i, i, 1)
        score_list[i] = LA.norm(y - X @ gamma_i, 2) ** 2 / n + alpha_i * LA.norm(gamma_i, 1)
    est_T = np.diag(score_list ** (-2))
    est_inverse_Sigma = est_T @ est_C
    return est_beta + (est_inverse_Sigma @ Q.T @ (Y - Q @ est_beta)) / n


def LDPE_statistics(est_beta, Y, Q, sigma):
    n, p = Q.shape
    est_C = np.zeros([p, p])
    score_list = np.zeros(p)
    for i in range(p):
        y = Q[:, i]
        X = np.delete(Q, i, axis=1)
        reg_i = LassoCV(cv=5, random_state=0).fit(X, y)
        gamma_i = reg_i.coef_
        alpha_i = reg_i.alpha_
        est_C[i] = np.insert(-gamma_i, i, 1)
        score_list[i] = LA.norm(y - X @ gamma_i, 2) ** 2 / n + alpha_i * LA.norm(gamma_i, 1)
    est_T = np.diag(score_list ** (-2))
    est_inverse_Sigma = est_T @ est_C
    LDPE_beta = est_beta + (est_inverse_Sigma @ Q.T @ (Y - Q @ est_beta)) / n
    Sigma = sigma ** 2 * est_inverse_Sigma @ (Q.T @ Q) @ est_inverse_Sigma.T / n
    return n ** (0.5) * LDPE_beta[0] / Sigma[0, 0] ** 0.5


def test_power_simulation_ldpe(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma, n):
    #     beta[0]=0
    rej_list = np.zeros(len(theta_candidate))
    for i in range(KK):
        np.random.seed(seed + i)
        ERR_temp = normal(mu, sigma, n)
        #         print(random.random())
        Q_temp = multivariate_normal(mean=mean1, cov=Sigma2, size=n)
        p = len(mean1)
        hyp_beta = np.zeros(p)
        hyp_beta[1:3] = 1
        for j in range(len(theta_candidate)):
            hyp_beta[0] = theta_candidate[j]
            Y_temp = ERR_temp + Q_temp @ hyp_beta
            reg_temp = LassoCV(cv=5, random_state=0).fit(Q_temp, Y_temp)
            temp_beta = reg_temp.coef_
            #             print(temp_beta[0])
            u_score = LDPE_statistics(temp_beta, Y_temp, Q_temp, sigma)
            p_value = 1 - 2 * (1 - stats.norm.cdf(abs(u_score)))
            if p_value > 1 - alpha:
                rej_list[j] += 1
    return rej_list


def find_KK(seed, alpha, can_KK, mu, sigma, mean1, Sigma, n):
    #     beta[0]=0
    rej_list = np.zeros(len(can_KK))
    for i in range(len(can_KK)):
        KK = can_KK[i]
        for j in range(KK):
            np.random.seed(seed + j)
            ERR_temp = normal(mu, sigma, n)
            Q_temp = multivariate_normal(mean=mean1, cov=Sigma, size=n)
            p = len(mean1)
            hyp_beta = np.zeros(p)
            hyp_beta[1:3] = 1
            Y_temp = ERR_temp + Q_temp @ hyp_beta
            reg_temp = LassoCV(cv=5, random_state=0).fit(Q_temp, Y_temp)
            temp_beta = reg_temp.coef_
            #             print(Y_temp[0])
            u_score = u_score_function(temp_beta, Y_temp, Q_temp, sigma)
            p_value = 1 - 2 * (1 - stats.norm.cdf(abs(u_score)))
            if p_value > 1 - alpha:
                rej_list[i] += 1
        rej_list[i] = rej_list[i] / KK

    return rej_list
