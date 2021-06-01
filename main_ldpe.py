import time
import warnings
import numpy as np
import func

# warnings.filterwarnings("ignore")
# warnings.simplefilter("default")
import pandas as pd
import openpyxl
import multiprocessing as mp
import numexpr as ne
from numpy.random import multivariate_normal
from numpy.random import normal
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsCV

# HERE: reset number of vml-threads
ne.set_vml_num_threads(8)


def main():
    seed = 10
    KK = 150
    alpha = 0.05
    theta_candidate = np.linspace(0, 0.05, 20)
    n = 100
    p = 100
    Sigma1 = 0.8 * np.ones([p, p])
    Sigma2 = np.zeros([p, p])
    rho = 0.6
    for i in range(p):
        for j in range(p):
            Sigma2[i, j] = rho ** abs(i - j)
    np.fill_diagonal(Sigma1, 1)
    mean1 = np.zeros(p)
    mu, sigma = 0, 0.1
    #
    pool = mp.Pool(4)
    np.random.seed(10)
    Q = multivariate_normal(mean=mean1, cov=Sigma2, size=n)
    ERR = normal(mu, sigma, n)
    beta = np.zeros(p)
    beta[1:3] = 1
    beta[0] = 0.03
    Y = ERR + Q @ beta

    reg_ldpe = LassoLarsCV(cv=5).fit(Q, Y)
    est_beta = reg_ldpe.coef_
    # lambda_desf = reg_ldpe.alpha_
    print(est_beta)
    start = time.time()
    para_ldpe_beta = func.parallel_LDPE_statistics(est_beta, Y, Q, sigma, pool)
    pool.close()
    pool.join()
    end = time.time()
    print('the running time is:' + str(end - start) + 's')
    start = time.time()
    ldpe_beta = func.LDPE_statistics(est_beta, Y, Q, sigma)
    end = time.time()
    print('the running time is:' + str(end - start) + 's')

    start = time.time()
    ldpe_beta_python = func.GLASSO_LDPE_statistics(est_beta, Y, Q, sigma)
    end = time.time()
    print('the running time is:' + str(end - start) + 's')

    print(para_ldpe_beta)
    print(ldpe_beta)
    print(ldpe_beta_python)

    # beta = np.zeros(p)
    # beta[1:3] = 1
    # beta[0] = 0
    # pool = mp.Pool(6)
    # start = time.time()
    # test_power_ldpe_p50 = func.parallel_test_power_simulation_ldpe(seed, 150, alpha, theta_candidate, mu, sigma, mean1,
    #                                                                Sigma2, n, pool)
    # end = time.time()
    # print('the running time is:' + str(end - start) + 's')
    # #
    # pool.close()
    # pool.join()
    # df = pd.DataFrame(test_power_ldpe_p50, columns=['LDPE'])
    # df.to_excel("test_power_ldpe_p100.xlsx")

    # p = 150
    # Sigma1 = 0.8 * np.ones([p, p])
    # Sigma2 = np.zeros([p, p])
    # rho = 0.6
    # for i in range(p):
    #     for j in range(p):
    #         Sigma2[i, j] = rho ** abs(i - j)
    # np.fill_diagonal(Sigma1, 1)
    # mean1 = np.zeros(p)
    # mu, sigma = 0, 0.1
    # test_power_ldpe_p150 = func.test_power_simulation_ldpe(seed, KK, alpha, theta_candidate, mu, sigma, mean1, Sigma2,
    #                                                        n)
    # df = pd.DataFrame(test_power_ldpe_p150, columns=['LDPE'])
    # df.to_excel("test_power_ldpe_p150.xlsx")


if __name__ == '__main__':
    main()
