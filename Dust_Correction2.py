"""
Author: Finlay Sime

Finds the best extinction value for correcting the interstellar reddening due to
interstellar dust.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from astropy.table import Table
import json

ms_data_path = 'main_sequence.dat'
fake_cluster_path = 'fake_cluster_clean.dat'
cluster_path = 'final_catalogue_M52.fits'
cluster_path_2sigma = 'final_catalogue_M52_2sigma.fits'

# Assuming data is in a structured format like CSV or similar
ms_data = pd.read_csv(ms_data_path, sep='\\s+', comment='#', engine='python')
fake_cluster_data = pd.read_csv(fake_cluster_path, sep='\\s+', comment='#', engine='python')
cluster_data = Table.read(cluster_path, format='fits')

# convert to numpy array indexing
ms_data = ms_data.values
fake_cluster_data = fake_cluster_data.values

####################################
# fake cluster data is organised as:
# ID U U_e B B_e V V_e U-B B-V

# main sequence data is organised as:
# SpT Teff logT BCv Mv logL B-V

# cluster data is organised as:
# thresh npix tnpix ... mag_v magerr_v mab_b magerr_b mag_u magerr_u ...
# [0]    [1]  [2]   ...
####################################

cluster_mag_v = np.array(cluster_data['mag_v'].tolist())
cluster_mag_b = np.array(cluster_data['mag_b'].tolist())
cluster_mag_u = np.array(cluster_data['mag_u'].tolist())
cluster_magerr_v = np.array(cluster_data['magerr_v'].tolist())
cluster_magerr_b = np.array(cluster_data['magerr_b'].tolist())
cluster_magerr_u = np.array(cluster_data['magerr_u'].tolist())

# filter out data points with None value in at least one band
remove_index = []
for i in range(0, len(cluster_mag_v)):
    if cluster_mag_v[i] is None:
        remove_index.append(i)
    elif cluster_mag_b[i] is None:
        remove_index.append(i)
    elif cluster_mag_u[i] is None:
        remove_index.append(i)
    elif cluster_magerr_v[i] is None:
        remove_index.append(i)
    elif cluster_magerr_b[i] is None:
        remove_index.append(i)
    elif cluster_magerr_u[i] is None:
        remove_index.append(i)
    else:
        continue

i = np.array(remove_index)
cluster_mag_v = np.delete(cluster_mag_v, i)
cluster_mag_b = np.delete(cluster_mag_b, i)
cluster_mag_u = np.delete(cluster_mag_u, i)
cluster_magerr_v = np.delete(cluster_magerr_v, i)
cluster_magerr_b = np.delete(cluster_magerr_b, i)
cluster_magerr_u = np.delete(cluster_magerr_u, i)

cluster_mag_v = cluster_mag_v.astype(float)
cluster_mag_b = cluster_mag_b.astype(float)
cluster_mag_u = cluster_mag_u.astype(float)
cluster_magerr_v = cluster_magerr_v.astype(float)
cluster_magerr_b = cluster_magerr_b.astype(float)
cluster_magerr_u = cluster_magerr_u.astype(float)

cluster_bv = cluster_mag_b - cluster_mag_v
cluster_ub = cluster_mag_u - cluster_mag_b
cluster_err_bv = np.sqrt(cluster_magerr_v**2 + cluster_magerr_b**2)
cluster_err_ub = np.sqrt(cluster_magerr_u**2 + cluster_magerr_b**2)

# filter out points with large uncertainties
remove_index = []
for i in range(0, len(cluster_err_ub)):
    if cluster_err_ub[i] >= 0.05:
        remove_index.append(i)
    elif cluster_err_bv[i] >= 0.05:
        remove_index.append(i)

j = np.array(remove_index)
cluster_ub = np.delete(cluster_ub, j)
cluster_bv = np.delete(cluster_bv, j)
cluster_mag_v = np.delete(cluster_mag_v, j)
cluster_magerr_v = np.delete(cluster_magerr_v, j)
cluster_err_ub = np.delete(cluster_err_ub, j)
cluster_err_bv = np.delete(cluster_err_bv, j)

# Calculate SNR to clean data
window_size = 2  # adjust based on data density
local_background = np.zeros_like(cluster_ub)
local_noise = np.zeros_like(cluster_ub)

for i in range(len(cluster_ub)):
    start = max(0, i - window_size // 2)
    end = min(len(cluster_ub), i + window_size // 2 + 1)

    local_background[i] = np.median(cluster_ub[start:end])
    local_noise[i] = np.median(cluster_err_ub[start:end])

# SNR = deviation from local background / local uncertainty
deviation = np.abs(cluster_ub - local_background)
SNR = deviation / np.sqrt(cluster_err_ub**2 + local_noise**2)

threshold = 3
cluster_ub = cluster_ub[SNR>threshold]
cluster_err_ub = cluster_err_ub[SNR>threshold]
cluster_bv = cluster_bv[SNR>threshold]
cluster_err_bv = cluster_err_bv[SNR>threshold]
cluster_mag_v = cluster_mag_v[SNR>threshold]
cluster_magerr_v = cluster_magerr_v[SNR>threshold]

# save list outputs for main sequence fitting
MainSequenceFittingList = list(set(zip(cluster_mag_v, cluster_magerr_v, cluster_bv, cluster_err_bv)))

with open("MainSequenceFittingList.txt", "w") as file:
    json.dump(MainSequenceFittingList, file)

# Values for R_V, A_U/A_V and A_B/A_V are taken from Cardelli 89
#A_V = np.arange(0, 2, 0.000001) # overall range
A_V = np.arange(1.1117, 1.1127, 0.0000001) # 1 sigma tolerance
#A_V = np.arange(1.0799007, 1.0799009, 0.0000001) # 2 sigma tolerance

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

# fitted interpolation
f1 = interpolate.interp1d(BV_sorted, UB_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')

x_min, x_max = min(BV_sorted), max(BV_sorted)
x = np.arange(x_min, x_max, 0.001)
y = f1(x)

# minimising chi-squared
chiSquared = []
for i in range(len(A_V)):
    dx = delx[i]  # ∆(B-V)
    dy = dely[i]  # ∆(U-B)

    # shifted cluster data for this A_V value
    shifted_BV = cluster_bv - dx
    shifted_UB = cluster_ub - dy

    # calculate chi-squared
    residuals = shifted_UB - f1(shifted_BV)
    chi = np.sum((residuals / cluster_err_ub)**2)
    chiSquared.append(chi)

chiSquared = np.array(chiSquared)

min_index_chi = np.argmin(chiSquared) # minimise chi-squared
optimal_A_V = A_V[min_index_chi] #find optimal value of A_V

delx = optimal_A_V / R_V # redefine x and y shifts for optimal A_V value
dely = optimal_A_V*(A_UdivA_V-A_BdivA_V)

# find the uncertainty on A_V #

# function to find the closest value to min chi-squared
def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

K = chiSquared[min_index_chi]+1
index1 = np.where(chiSquared == closest(chiSquared, K))

K = 2*optimal_A_V - A_V[index1]
index2 = np.where(A_V == closest(A_V, K))

# check errors are equal
lower_err = optimal_A_V - A_V[index1]
upper_err = A_V[index2] - optimal_A_V

# error finding of optimal_A_V
if lower_err == upper_err:
    optimal_A_V_err = np.mean([upper_err, upper_err])
    print(fr'Optimal $A_V$ error: {optimal_A_V_err}')
elif lower_err != upper_err:
    print(f'Error difference: {lower_err} - {upper_err}')
    optimal_A_V_err = np.mean([lower_err, upper_err])
else:
    print('Error finding the error on the optimal extinction co-efficient')

# plot A_V plot and colour-colour plot side by side
plt.errorbar(cluster_bv, cluster_ub,
             yerr=cluster_err_ub, capsize=5, fmt='o', elinewidth=1, capthick=1,
             color='green', label=f'Observed Cluster Data', markersize=1, zorder=1)
plt.errorbar(cluster_bv-delx, cluster_ub-dely,
             yerr=cluster_err_ub, capsize=5, fmt='o', elinewidth=1, capthick=1,
             color='red', label=fr'Dereddened Cluster Data, $A_V$={optimal_A_V:.3f}±{optimal_A_V_err:.3f}',
                    markersize=1, zorder=2)
plt.plot(x, y)
plt.scatter(BVint, UBint, color='blue', marker='o', label='Intrinsic Data', s=8, zorder=3)
plt.xlabel('(B-V)')
plt.ylabel('(U-B)')
plt.gca().invert_yaxis()
plt.legend()
plt.show()

plt.plot(A_V, chiSquared)
plt.errorbar(A_V[min_index_chi], chiSquared[min_index_chi], xerr=lower_err,
                   capsize=5, fmt='o', elinewidth=1, capthick=1, label=r'Minimum $\chi^2$')
plt.xlabel('$A_{V}$ values')
plt.ylabel(r'$\chi^{2}$')
plt.legend()
plt.show()

fig, ax = plt.subplot_mosaic([['data', 'A_V']], figsize=(14, 5))

ax['data'].errorbar(cluster_bv, cluster_ub,
             yerr=cluster_err_ub, capsize=5, fmt='o', elinewidth=1, capthick=1,
             color='green', label=f'Observed Cluster Data', markersize=1, zorder=1)
ax['data'].errorbar(cluster_bv-delx, cluster_ub-dely,
             yerr=cluster_err_ub, capsize=5, fmt='o', elinewidth=1, capthick=1,
             color='red', label=fr'Dereddened Cluster Data, $A_V$={optimal_A_V:.3f}±{optimal_A_V_err:.3f}',
                    markersize=1, zorder=2)
ax['data'].plot(x, y)
ax['data'].scatter(BVint, UBint, color='blue', marker='o', label='Intrinsic Data', s=8, zorder=3)
ax['data'].set_xlabel('(B-V)')
ax['data'].set_ylabel('(U-B)')
ax['data'].legend()
ax['data'].invert_yaxis()
ax['A_V'].plot(A_V, chiSquared)
ax['A_V'].errorbar(A_V[min_index_chi], chiSquared[min_index_chi], xerr=lower_err,
                   capsize=5, fmt='o', elinewidth=1, capthick=1, label=r'Minimum $\chi^2$')
ax['A_V'].set_xlabel('$A_{V}$ values')
ax['A_V'].set_ylabel(r'$\chi^{2}$')
ax['A_V'].legend()
plt.show()
