import time
import warnings

import numpy as np
from numpy.random import multivariate_normal
from numpy.random import normal
from scipy import stats
from sklearn.linear_model import LassoCV, Lasso
from sklearn.covariance import GraphicalLassoCV
import multiprocessing as mp
from functools import partial
from sklearn.covariance import MinCovDet
from sklearn.datasets import make_gaussian_quantiles

warnings.filterwarnings("ignore")
# warnings.simplefilter("default")
from numpy import linalg as LA


def deco_score_function(est_beta, Y, Q, sigma):
    n, p = Q.shape
    X = Q[:, 1:]
    Z = Q[:, 0]
    #     reg_desf = LassoCV(cv=5, random_state=0).fit(Q, Y)
    #     est_beta = reg_desf.coef_
    #     lambda_desf = reg_desf.alpha_
    # est_theta = est_beta[0]
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

    # est_theta = est_beta[0]
    est_gamma = est_beta[1:]
    reg_w = LassoCV(cv=5, random_state=0).fit(X, Z)
    est_w = reg_w.coef_

    desf = -(1 / ((sigma ** 2) * n)) * np.dot(Y - X @ est_gamma, Z - X @ est_w)
    Z_matrix = np.repeat(np.reshape(Z, [n, 1]), p - 1, axis=1)  # ?
    I = 1 / ((sigma ** 2) * n) * (np.sum(Z ** 2) - np.sum(np.multiply(X, Z_matrix), axis=0) @ est_w)
    u_score = n ** 0.5 * desf * I ** (-0.5)
    return u_score


def u_score_CI(est_beta, Y, Q, alpha, sigma):
    n, p = Q.shape
    X = Q[:, 1:]
    Z = Q[:, 0]
    #     reg_desf = LassoCV(cv=5, random_state=0).fit(Q, Y)
    #     est_beta = reg_desf.coef_
    #     lambda_desf = reg_desf.alpha_

    # est_theta = est_beta[0]
    est_gamma = est_beta[1:]
    reg_w = LassoCV(cv=5, random_state=0).fit(X, Z)
    est_w = reg_w.coef_
    # desf = -(1 / ((sigma ** 2) * n)) * np.dot(Y - X @ est_gamma, Z - X @ est_w)
    Z_matrix = np.repeat(np.reshape(Z, [n, 1]), p - 1, axis=1)  # ?
    I_desf = 1 / ((sigma ** 2) * n) * (np.sum(Z ** 2) - np.sum(np.multiply(X, Z_matrix), axis=0) @ est_w)
    q_normal = stats.norm.ppf(1 - alpha / 2)
    # print(q_normal)
    sf_ci = 2 * I_desf ** (-0.5) * n ** (-.5) * q_normal
    return sf_ci


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

    print(rej_list / KK)

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
    est_LDPE_beta = est_beta + (est_inverse_Sigma @ Q.T @ (Y - Q @ est_beta)) / n
    Sigma = sigma ** 2 * est_inverse_Sigma @ (Q.T @ Q) @ est_inverse_Sigma.T / n
    return n ** 0.5 * est_LDPE_beta[0] / Sigma[0, 0] ** 0.5


def GLASSO_LDPE_statistics(est_beta, Y, Q, sigma):
    n, p = Q.shape
    est_C = np.zeros([p, p])
    score_list = np.zeros(p)
    cov = GraphicalLassoCV().fit(Q)
    est_inverse_Sigma = cov.precision_
    est_LDPE_beta = est_beta + (est_inverse_Sigma @ Q.T @ (Y - Q @ est_beta)) / n
    Sigma = sigma ** 2 * est_inverse_Sigma @ (Q.T @ Q) @ est_inverse_Sigma.T / n
    return n ** 0.5 * est_LDPE_beta[0] / Sigma[0, 0] ** 0.5


def LDPE_CI(est_beta, Y, Q, sigma):
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
    est_LDPE_beta = est_beta + (est_inverse_Sigma @ Q.T @ (Y - Q @ est_beta)) / n
    Sigma = sigma ** 2 * est_inverse_Sigma @ (Q.T @ Q) @ est_inverse_Sigma.T / n
    return n ** 0.5 * est_LDPE_beta[0] / Sigma[0, 0] ** 0.5


def test_power_simulation_ldpe(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma, n):
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
            u_score = LDPE_statistics(temp_beta, Y_temp, Q_temp, sigma)
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
            #             print(temp_beta[0])
            u_score = LDPE_statistics(temp_beta, Y_temp, Q_temp, sigma)
            p_value = 1 - 2 * (1 - stats.norm.cdf(abs(u_score)))
            if p_value > 1 - alpha:
                rej_list[j] += 1
    return rej_list / KK


def parallel_test_power_simulation_ldpe(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma, n, pool):
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
            start = time.time()
            u_score = parallel_LDPE_statistics(temp_beta, Y_temp, Q_temp, sigma, pool)
            end = time.time()
            print('the running time is:' + str(end - start) + 's')
            p_value = 1 - 2 * (1 - stats.norm.cdf(abs(u_score)))
            if p_value > 1 - alpha:
                rej_list[j] += 1
    lambda_list = lambda_list / 50
    print(rej_list / 50)

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
            #             print(temp_beta[0])
            u_score = parallel_LDPE_statistics(temp_beta, Y_temp, Q_temp, sigma, pool)
            p_value = 1 - 2 * (1 - stats.norm.cdf(abs(u_score)))
            if p_value > 1 - alpha:
                rej_list[j] += 1
    return rej_list / KK


def test_power_simulation_ldpe_graphical(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma, n):
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
            u_score = GLASSO_LDPE_statistics(temp_beta, Y_temp, Q_temp, sigma)
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
            #             print(temp_beta[0])
            u_score = GLASSO_LDPE_statistics(temp_beta, Y_temp, Q_temp, sigma)
            p_value = 1 - 2 * (1 - stats.norm.cdf(abs(u_score)))
            if p_value > 1 - alpha:
                rej_list[j] += 1
    return rej_list / KK


def test_power_simulation_ldpe_graphical(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma, n):
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
            u_score = GLASSO_LDPE_statistics(temp_beta, Y_temp, Q_temp, sigma)
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
            #             print(temp_beta[0])
            u_score = GLASSO_LDPE_statistics(temp_beta, Y_temp, Q_temp, sigma)
            p_value = 1 - 2 * (1 - stats.norm.cdf(abs(u_score)))
            if p_value > 1 - alpha:
                rej_list[j] += 1
    return rej_list / KK



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


def rates_type_one(data, KK):
    rej_rate = 0
    for i in range(KK):
        # data[i]
        p_value = 1 - 2 * (1 - stats.norm.cdf(abs(data[i])))
        if p_value <= 0.90:  # when accepting
            rej_rate += 1
    #     print("the rate of rejection is {:g}".format(1-rej_rate/KK))
    return 1 - rej_rate / KK


def i_iteration_LDPE(Q, i):
    n, p = Q.shape
    y = Q[:, i]
    X = np.delete(Q, i, axis=1)
    reg_i = LassoCV(cv=5, random_state=0).fit(X, y)
    gamma_i = reg_i.coef_
    alpha_i = reg_i.alpha_
    est_C_i = np.insert(-gamma_i, i, 1)
    score_i = LA.norm(y - X @ gamma_i, 2) ** 2 / n + alpha_i * LA.norm(gamma_i, 1)

    return score_i, est_C_i


def parallel_LDPE_statistics(est_beta, Y, Q, sigma, pool):
    n, p = Q.shape
    est_C = np.zeros([p, p])
    # score_list = np.zeros(p)
    parallel_function = partial(i_iteration_LDPE, Q)
    score_list, est_C = zip(*pool.map(parallel_function, [i for i in range(p)]))
    # print(score_list)
    score_list = np.array(score_list)
    est_C = np.array(est_C)
    # for i in range(p):
    #     y = Q[:, i]
    #     X = np.delete(Q, i, axis=1)
    #     reg_i = LassoCV(cv=5, random_state=0).fit(X, y)
    #     gamma_i = reg_i.coef_
    #     alpha_i = reg_i.alpha_

    #     score_list[i] = LA.norm(y - X @ gamma_i, 2) ** 2 / n + alpha_i * LA.norm(gamma_i, 1)
    # est_C = score_list[:, 1]
    est_T = np.diag(score_list ** (-2))
    est_inverse_Sigma = est_T @ est_C
    est_LDPE_beta = est_beta + (est_inverse_Sigma @ Q.T @ (Y - Q @ est_beta)) / n
    Sigma = sigma ** 2 * est_inverse_Sigma @ (Q.T @ Q) @ est_inverse_Sigma.T / n
    return n ** 0.5 * est_LDPE_beta[0] / Sigma[0, 0] ** 0.5


def av_rates_desf(seed, KK, alpha, n, p, s, rho):
    rej_rate = 0
    Sigma2 = np.zeros([p, p])
    # rho = 0.6
    for i in range(p):
        for j in range(p):
            Sigma2[i, j] = rho ** abs(i - j)
    # np.fill_diagonal(Sigma1, 1)
    mean1 = np.zeros(p)
    mu, sigma = 0, 0.1
    beta = np.zeros(p)
    beta[1:s + 1] = 1
    beta[0] = 0
    # print(beta)
    cv_lambda = 0
    for i in range(50):
        np.random.seed(seed + i)
        ERR = normal(mu, sigma, n)
        Q = multivariate_normal(mean=mean1, cov=Sigma2, size=n)
        Y = ERR + Q @ beta
        reg_desf = LassoCV(cv=5, random_state=0).fit(Q, Y)
        est_beta = reg_desf.coef_
        cv_lambda += reg_desf.alpha_
        uscore_i = u_score_function(est_beta, Y, Q, sigma)
        p_value = 1 - 2 * (1 - stats.norm.cdf(abs(uscore_i)))
        if p_value > 1 - alpha:
            rej_rate += 1
    cv_lambda = cv_lambda / 50

    for i in range(50, KK):
        np.random.seed(seed + i)
        ERR = normal(mu, sigma, n)
        Q = multivariate_normal(mean=mean1, cov=Sigma2, size=n)
        Y = ERR + Q @ beta
        reg_desf = Lasso(alpha=cv_lambda).fit(Q, Y)
        est_beta = reg_desf.coef_
        # cv_lambda += reg_desf.alpha_
        uscore_i = u_score_function(est_beta, Y, Q, sigma)
        # p_value = 1 - stats.norm.cdf(abs(uscore_i))
        p_value = 1 - 2 * (1 - stats.norm.cdf(abs(uscore_i)))
        if p_value > 1 - alpha:
            rej_rate += 1
    return rej_rate / KK


def av_ci_desf(seed, KK, alpha, n, p, s, rho):
    av_ci = 0
    Sigma2 = np.zeros([p, p])
    # rho = 0.6
    for i in range(p):
        for j in range(p):
            Sigma2[i, j] = rho ** abs(i - j)
    # np.fill_diagonal(Sigma1, 1)
    mean1 = np.zeros(p)
    mu, sigma = 0, 0.1
    beta = np.zeros(p)
    beta[1:s + 1] = 1
    beta[0] = 0
    # print(beta)
    cv_lambda = 0
    for i in range(KK):
        np.random.seed(seed + i)
        ERR = normal(mu, sigma, n)
        Q = multivariate_normal(mean=mean1, cov=Sigma2, size=n)
        Y = ERR + Q @ beta
        reg_desf = LassoCV(cv=5, random_state=0).fit(Q, Y)
        est_beta = reg_desf.coef_
        cv_lambda += reg_desf.alpha_
        score_ci = u_score_CI(est_beta, Y, Q, alpha, sigma)
        # print(score_ci)
        av_ci += score_ci

    # cv_lambda = cv_lambda / 50
    #
    # for i in range(50, KK):
    #     np.random.seed(seed + i)
    #     ERR = normal(mu, sigma, n)
    #     Q = multivariate_normal(mean=mean1, cov=Sigma2, size=n)
    #     Y = ERR + Q @ beta
    #     reg_desf = Lasso(alpha=cv_lambda).fit(Q, Y)
    #     est_beta = reg_desf.coef_
    #     # cv_lambda += reg_desf.alpha_
    #     score_ci = u_score_CI(est_beta, Y, Q, alpha, sigma)
    #     # print(score_ci)
    #     av_ci += score_ci

    return av_ci / KK


def av_rates_ldpe(seed, KK, alpha, n, p, s, rho, pool):
    rej_rate = 0
    Sigma2 = np.zeros([p, p])
    # rho = 0.6
    for i in range(p):
        for j in range(p):
            Sigma2[i, j] = rho ** abs(i - j)
    # np.fill_diagonal(Sigma1, 1)
    mean1 = np.zeros(p)
    mu, sigma = 0, 0.1
    beta = np.zeros(p)
    beta[1:s + 1] = 1
    beta[0] = 0
    # print(beta)
    cv_lambda = 0
    for i in range(50):
        np.random.seed(seed + i)
        ERR = normal(mu, sigma, n)
        Q = multivariate_normal(mean=mean1, cov=Sigma2, size=n)
        Y = ERR + Q @ beta
        reg_desf = LassoCV(cv=5, random_state=0).fit(Q, Y)
        est_beta = reg_desf.coef_
        cv_lambda += reg_desf.alpha_
        uscore_i = parallel_LDPE_statistics(est_beta, Y, Q, sigma, pool)
        p_value = 1 - 2 * (1 - stats.norm.cdf(abs(uscore_i)))
        if p_value > 1 - alpha:
            rej_rate += 1
    cv_lambda = cv_lambda / 50

    for i in range(50, KK):
        np.random.seed(seed + i)
        ERR = normal(mu, sigma, n)
        Q = multivariate_normal(mean=mean1, cov=Sigma2, size=n)
        Y = ERR + Q @ beta
        reg_desf = Lasso(alpha=cv_lambda).fit(Q, Y)
        est_beta = reg_desf.coef_
        # cv_lambda += reg_desf.alpha_
        uscore_i = parallel_LDPE_statistics(est_beta, Y, Q, sigma, pool)
        # p_value = 1 - stats.norm.cdf(abs(uscore_i))
        p_value = 1 - 2 * (1 - stats.norm.cdf(abs(uscore_i)))
        if p_value > 1 - alpha:
            rej_rate += 1

    print(rej_rate)

    return rej_rate / KK
