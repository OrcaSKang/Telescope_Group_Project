"""
Author: Finlay Sime

Finds the best extinction value for correcting the interstellar reddening due to
interstellar dust.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

ms_data_path = 'main_sequence.dat'
fake_cluster_path = 'fake_cluster_clean.dat'

# Assuming data is in a structured format like CSV or similar
ms_data = pd.read_csv(ms_data_path, sep='\\s+', comment='#', engine='python')
fake_cluster_data = pd.read_csv(fake_cluster_path, sep='\\s+', comment='#', engine='python')

# convert to numpy array indexing
ms_data = ms_data.values
fake_cluster_data = fake_cluster_data.values

####################################
# fake cluster data is organised as:
# ID U U_e B B_e V V_e U-B B-V

# main sequence data is organised as:
# SpT Teff logT BCv Mv logL B-V
####################################

# Values for R_V, A_U/A_V and A_B/A_V are taken from Cardelli 89
A_V = np.arange(0.96, 1, 0.000001)
R_V = 3.1

A_V = np.array(A_V)

A_UdivA_V = 1.569
A_BdivA_V = 1.337

# reddening vector components
delx = A_V / R_V
dely = A_V*(A_UdivA_V-A_BdivA_V)

UBerr = np.sqrt(fake_cluster_data[:,2]**2 + fake_cluster_data[:,4]**2)
BVerr = np.sqrt(fake_cluster_data[:,4]**2 + fake_cluster_data[:,6]**2)

# Intrinsic main sequence colours are taken from https://www.stsci.edu/~inr/intrins.html
# (U-B) intrinsic
UBint = [-1.08, -1.00, -0.95, -0.88, -0.81, -0.72, -0.68, -0.65, -0.63,
         -0.61, -0.58, -0.49, -0.43, -0.40, -0.36, -0.27, -0.18, -0.10,
         -0.02, 0.01, 0.05, 0.08, 0.09, 0.09, 0.10, 0.10, 0.09, 0.08, 0.03, 0.00, 0.00,
         -0.02, 0.02, 0.06, 0.09, 0.12, 0.20, 0.30, 0.44, 0.48, 0.67,
         0.73, 1.00, 1.06, 1.21, 1.23, 1.18, 1.15, 1.17, 1.07]
# (B-V) intrinsic
BVint = [-0.30, -0.28, -0.26, -0.25, -0.24, -0.22, -0.20, -0.19, -0.18,
         -0.17, -0.16, -0.14, -0.13, -0.12, -0.11, -0.09, -0.07, -0.04,
         -0.01, 0.02, 0.05, 0.08, 0.12, 0.15, 0.17, 0.20, 0.27, 0.30,
         0.32, 0.34, 0.35, 0.45, 0.53, 0.60, 0.63, 0.65, 0.68, 0.74, 0.81,
         0.86, 0.92, 0.95, 1.00, 1.15, 1.33, 1.37, 1.47, 1.47, 1.50, 1.52]

# merge intrinsic data sets, remove duplicates and sort
int_colour = list(set(zip(BVint, UBint)))
int_colour.sort(key=lambda x: x[1])

BV_sorted = [item[0] for item in int_colour]
UB_sorted = [item[1] for item in int_colour]

# Corresponding spectral type
SpecT = ['B0.0', 'B0.05', 'B1.0', 'B1.5', 'B2.0', 'B2.5', 'B3.0', 'B3.5', 'B4.0', 'B4.5',
         'B5.5', 'B6.0', 'B7.0', 'B7.5', 'B8.0', 'B8.5', 'B9.0', 'B9.5', 'A0.0', 'A1.0',
         'A2.0', 'A3.0', 'A4.0', 'A5.0', 'A6.0', 'A7.0', 'A8.0', 'A9.0', 'F0.0', 'F1.0',
         'F2.0', 'F5.0', 'F8.0', 'G0.0', 'G2.0', 'G3.0', 'G5.0', 'G8.0', 'K0.0', 'K1.0', 'K2.0',
         'K3.0', 'K4.0', 'K5.0', 'K7.0', 'M0.0', 'M1.0', 'M2.0', 'M3.0', 'M4.0']

f1 = interpolate.interp1d(BV_sorted, UB_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')

x_min, x_max = min(BV_sorted), max(BV_sorted)
x = np.arange(x_min, x_max, 0.01)

y = f1(x)

chiSquared = []
for i in range(len(A_V)):
    dx = delx[i]  # ∆(B-V)
    dy = dely[i]  # ∆(U-B)

    # shifted cluster data for this A_V value
    shifted_BV = fake_cluster_data[:,-1] - dx
    shifted_UB = fake_cluster_data[:,-2] - dy

    # calculate chi-squared
    residuals = shifted_UB - f1(shifted_BV)
    chi = np.sum((residuals / UBerr)**2)
    chiSquared.append(chi)

chiSquared = np.array(chiSquared)

min_index_chi = np.argmin(chiSquared) # minimise chi-squared
optimal_A_V = A_V[min_index_chi] #find optimal value of A_V

delx = optimal_A_V / R_V # redefine x and y shifts for optimal A_V value
dely = optimal_A_V*(A_UdivA_V-A_BdivA_V)

# find the uncertainty on A_V

# function to find closest value to min chi-squared
def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

K = chiSquared[min_index_chi]+1
index1 = np.where(chiSquared == closest(chiSquared, K))

K = 2*optimal_A_V - A_V[index1]
index2 = np.where(A_V == closest(A_V, K))

# check errors are equal
lower_err = optimal_A_V - A_V[index1]
upper_err = A_V[index2] - optimal_A_V

# error finding
if lower_err == upper_err:
    optimal_A_V_err = upper_err
elif lower_err != upper_err:
    print(f'Error difference: {lower_err} - {upper_err}')
    optimal_A_V_err = np.mean([lower_err, upper_err])
else:
    print('How is it different')

# plot A_V plot and colour-colour plot side by side
fig, ax = plt.subplot_mosaic([['data', 'A_V']], figsize=(12, 5))
ax['data'].errorbar(fake_cluster_data[:,-1], fake_cluster_data[:,-2],
             yerr=UBerr, capsize=5, fmt='o', elinewidth=1, capthick=1, # for not ignore x errors
             color='green', label=f'Observed Cluster Data', markersize=1, zorder=1)
ax['data'].errorbar(fake_cluster_data[:,-1]-delx, fake_cluster_data[:,-2]-dely,
             yerr=UBerr, capsize=5, fmt='o', elinewidth=1, capthick=1,
             color='red', label=fr'Dereddened Cluster Data, $A_V$={optimal_A_V:.3f}±{optimal_A_V_err[0]:.3f}',
                    markersize=1, zorder=2)
ax['data'].plot(x, y)
ax['data'].scatter(BVint, UBint, color='blue', marker='o', label='Intrinsic Data', s=8, zorder=3)
ax['data'].set_xlabel('(B-V)')
ax['data'].set_ylabel('(U-B)')
ax['data'].legend()
ax['data'].invert_yaxis()
ax['A_V'].plot(A_V, chiSquared)
ax['A_V'].scatter(A_V[min_index_chi], chiSquared[min_index_chi], c='r', label=r'Minimum $\chi^2$')
ax['A_V'].scatter(A_V[index1], chiSquared[index1], c='g', label=fr'1+$\chi^2$')
ax['A_V'].scatter(A_V[index2], chiSquared[index2], c='g')
ax['A_V'].set_xlabel('$A_{V}$ values')
ax['A_V'].set_ylabel(r'$\chi^{2}$')
ax['A_V'].legend()
plt.show()