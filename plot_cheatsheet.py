import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df1 = pd.read_excel('test_power_p50.xlsx')
df2 = pd.read_excel('test_power_p100.xlsx')
df3 = pd.read_excel('test_power_p150.xlsx')
df4 = pd.read_excel('test_power_ldpe_p100.xlsx')
df5 = pd.read_excel('test_power_p50_gldpe.xlsx')
theta_candidate = np.linspace(0, 0.05, 20)

sns.set_style('darkgrid')
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('font', size=13)
alpha = 0.05

plt.plot(theta_candidate, df3['DeSF'], label='DeSF')
# plt.plot(theta_candidate, df1['LDPE'], label='LDPE')
plt.plot(theta_candidate, alpha * np.ones(len(theta_candidate)), 'r--', label='alpha = 0.05')
plt.ylabel('Test Power (Empirical Rejection Rates)')
plt.ylabel('Test Power (Empirical Rejection Rates)')
plt.legend()
plt.title(' n  = 100, p = 150 ')
plt.savefig('test_power_p150')
plt.show()

plt.figure(figsize=(7.5, 6), dpi=80)
can_s = np.arange(2, 11)
s_rate_table_p50_1 = np.abs(np.array([0.04, 0.052, 0.052, 0.048, 0.04, 0.032, 0.036, 0.032, 0.032]))
s_rate_table_p50_2 = np.abs(np.array([0.054, 0.058, 0.058, 0.056, 0.05, 0.046, 0.044, 0.04, 0.036]))
s_rate_table_p50_3 = np.abs(np.array([0.055, 0.058, 0.057, 0.056, 0.054, 0.051, 0.051, 0.048, 0.046]))
alpha_line = np.ones(9) * 0.05
plt.plot(can_s, s_rate_table_p50_1, label='K = 250')
plt.plot(can_s, s_rate_table_p50_2, label='K = 500')
plt.plot(can_s, s_rate_table_p50_3, label='K = 1000')
plt.plot(can_s, alpha_line, 'r--')
plt.ylim(0, 0.1)
plt.ylabel('Test Validity (Average Type One Error Rate)')
plt.xlabel('s (number of non-zero elements in beta)')
plt.title('p = 50, n = 100')
plt.legend()
plt.savefig('av_error_against_s')
plt.show()

plt.plot(theta_candidate, df5['LDPE'], label='DeSF')
plt.plot(theta_candidate, df1['LDPE'], label='LDPE')
plt.plot(theta_candidate, alpha * np.ones(len(theta_candidate)), 'r--', label='alpha = 0.05')
plt.ylabel('Test Power (Empirical Rejection Rates)')
plt.legend()
plt.title(' n  = 100, p = 50 ')
plt.savefig('test_power_p50_3mthds')
plt.show()



plt.plot(theta_candidate, df1['DeSF'], label='DeSF')
plt.plot(theta_candidate, df1['LDPE'], label='LDPE (Node LASSO)')
plt.plot(theta_candidate, df5['DeSF'], label='LDPE (Graphical LASSO)')
plt.plot(theta_candidate, alpha * np.ones(len(theta_candidate)), 'r--', label='alpha = 0.05')
plt.ylabel('Test Power (Empirical Rejection Rates)')
plt.legend()
plt.title(' n  = 100, p = 50 ')
plt.savefig('test_power_p50_3mthds')
plt.show()
