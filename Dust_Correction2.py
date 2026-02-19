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

ms_data_path = 'main_sequence.dat'
fake_cluster_path = 'fake_cluster_clean.dat'
cluster_path = 'final_catalogue_M52.fits'
cluster_path_2sigma = 'final_catalogue_M52_2sigma.fits'
cluster_path_2 = 'final_catalogue_M52_2.fits'

# Assuming data is in a structured format like CSV or similar
ms_data = pd.read_csv(ms_data_path, sep='\\s+', comment='#', engine='python')
fake_cluster_data = pd.read_csv(fake_cluster_path, sep='\\s+', comment='#', engine='python')
cluster_data = Table.read(cluster_path, format='fits')

# convert to numpy array indexing
ms_data = ms_data.values
fake_cluster_data = fake_cluster_data.values

# filter out points beneath a SNR threshold
snr_threshold = 3.0
ok = ((cluster_data['flux_u']/cluster_data['fluxerr_u'] > snr_threshold) &
      (cluster_data['flux_v']/cluster_data['fluxerr_v'] > snr_threshold) &
      (cluster_data['flux_b']/cluster_data['fluxerr_b'] > snr_threshold))
cluster_data = cluster_data[ok]

cluster_mag_v = np.array(cluster_data['mag_v'].tolist())
cluster_mag_b = np.array(cluster_data['mag_b'].tolist())
cluster_mag_u = np.array(cluster_data['mag_u'].tolist())
cluster_magerr_v = np.array(cluster_data['magerr_v'].tolist())
cluster_magerr_b = np.array(cluster_data['magerr_b'].tolist())
cluster_magerr_u = np.array(cluster_data['magerr_u'].tolist())

cluster_mag_v = cluster_mag_v.astype(float)
cluster_mag_b = cluster_mag_b.astype(float)
cluster_mag_u = cluster_mag_u.astype(float)
cluster_magerr_v = cluster_magerr_v.astype(float)
cluster_magerr_b = cluster_magerr_b.astype(float)
cluster_magerr_u = cluster_magerr_u.astype(float)

# get colours
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
cluster_mag_b = np.delete(cluster_mag_b, j)
cluster_mag_u = np.delete(cluster_mag_u, j)
cluster_magerr_u = np.delete(cluster_magerr_u, j)
cluster_magerr_b = np.delete(cluster_magerr_b, j)

# Intrinsic main sequence colours are taken from https://www.stsci.edu/~inr/intrins.html
# (U-B) intrinsic
UBint = [-1.08, -1.00, -0.95, -0.88, -0.81, -0.72, -0.68, -0.65, -0.63, -0.61, -0.58, -0.49, -0.43, -0.40, -0.36, -0.27, -0.18, -0.10, -0.02, 0.01, 0.05, 0.08, 0.09, 0.09, 0.10, 0.10, 0.09, 0.08, 0.03, 0.00, 0.00, -0.02, 0.02, 0.06, 0.09, 0.12, 0.20, 0.30, 0.44, 0.48, 0.67, 0.73, 1.00, 1.06, 1.21, 1.23, 1.18, 1.15, 1.17, 1.07]
# (B-V) intrinsic
BVint = [-0.30, -0.28, -0.26, -0.25, -0.24, -0.22, -0.20, -0.19, -0.18, -0.17, -0.16, -0.14, -0.13, -0.12, -0.11, -0.09, -0.07, -0.04, -0.01, 0.02, 0.05, 0.08, 0.12, 0.15, 0.17, 0.20, 0.27, 0.30, 0.32, 0.34, 0.35, 0.45, 0.53, 0.60, 0.63, 0.65, 0.68, 0.74, 0.81, 0.86, 0.92, 0.95, 1.00, 1.15, 1.33, 1.37, 1.47, 1.47, 1.50, 1.52]

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

# function to calculate chi-sqaured for A_V values
def ComputeChiSquared(delx, dely, bv_data, ub_data, function):
    chiSquared = []

    for i in range(len(A_V)):
        dx = delx[i]  # ∆(B-V)
        dy = dely[i]  # ∆(U-B)

        # shifted cluster data for this A_V value
        shifted_BV = bv_data-dx
        shifted_UB = ub_data-dy

        expected_ub = function(shifted_BV)

        # calculate chi-squared
        residuals = shifted_UB-expected_ub
        chi = np.sum((residuals**2)/(expected_ub+1e-20))
        chiSquared.append(chi)

    return chiSquared

# Values for R_V, A_U/A_V and A_B/A_V are taken from Cardelli 89
A_V = np.arange(0, 5, 0.0001) # overall range
R_V = 3.1
A_V = np.array(A_V)
A_UdivA_V = 1.569
A_BdivA_V = 1.337
# reddening vector components
delx = A_V / R_V
dely = A_V*(A_UdivA_V-A_BdivA_V)

chiSquaredOriginal = ComputeChiSquared(delx, dely, cluster_bv, cluster_ub, f1)

min_index_chi = np.argmin(chiSquaredOriginal) # minimise chi-squared
optimal_A_V = A_V[min_index_chi] #find optimal value of A_V

# implement Monte Carlo Method to calculate the uncertainty of A_V

A_V_perturbed_values = []
N = 1000
print(f"Running Monte Carlo simulation for {N} iterations...")
for iteration in range(N):
    print(f"    Iteration {iteration}/{N}")
    perturbed_mag_v = np.random.normal(cluster_mag_v, cluster_magerr_v)
    perturbed_mag_b = np.random.normal(cluster_mag_b, cluster_magerr_b)
    pertubred_mag_u = np.random.normal(cluster_mag_u, cluster_magerr_u)

    perturbed_bv = perturbed_mag_b - perturbed_mag_v
    perturbed_ub = pertubred_mag_u - perturbed_mag_b

    chiSquaredPerturbed = ComputeChiSquared(delx, dely, perturbed_bv, perturbed_ub, f1)
    min_index = np.argmin(chiSquaredPerturbed)
    A_V_perturbed_values.append(A_V[min_index])

A_V_perturbed_values = np.array(A_V_perturbed_values)

A_V_mean = np.mean(A_V_perturbed_values)
A_V_std = np.std(A_V_perturbed_values)
A_V_median = np.median(A_V_perturbed_values)

percentiles = np.percentile(A_V_perturbed_values, [16, 50, 84])
lower_uncertainty = percentiles[1] - percentiles[0]
upper_uncertainty = percentiles[2] - percentiles[1]

print("\n=== Monte Carlo Results ===")
print(f"Optimal A_V from original data: {optimal_A_V:.4f}")
print(f"Mean A_V from MC: {A_V_mean:.4f}")
print(f"Median A_V from MC: {A_V_median:.4f}")
print(f"Standard deviation: {A_V_std:.4f}")
print(f"68% confidence interval: [{percentiles[0]:.4f}, {percentiles[2]:.4f}]")
print(f"Asymmetric uncertainties: -{lower_uncertainty:.4f}, +{upper_uncertainty:.4f}")

A_V_uncertainty = A_V_std

delx = optimal_A_V / R_V # redefine x and y shifts for optimal A_V value
dely = optimal_A_V*(A_UdivA_V-A_BdivA_V)

# plot colour-colour plot with original and dereddened data
plt.errorbar(cluster_bv, cluster_ub,
             yerr=cluster_err_ub, capsize=5, fmt='o', elinewidth=1, capthick=1,
             color='green', label=f'Observed Cluster Data', markersize=1, zorder=1)
plt.errorbar(cluster_bv-delx, cluster_ub-dely,
             yerr=cluster_err_ub, capsize=5, fmt='o', elinewidth=1, capthick=1,
             color='red', label=fr'Dereddened Cluster Data, $A_V$={optimal_A_V:.3f}±{A_V_uncertainty:.3f}',
                    markersize=1, zorder=2)
plt.plot(x, y)
plt.scatter(BVint, UBint, color='blue', marker='o', label='Intrinsic Data', s=8, zorder=3)
plt.xlabel('(B-V)')
plt.ylabel('(U-B)')
plt.gca().invert_yaxis()
plt.legend()
plt.show()

# plot the chi-squared value for each A_V
plt.plot(A_V, chiSquaredOriginal)
plt.errorbar(A_V[min_index_chi], chiSquaredOriginal[min_index_chi],
                   capsize=5, fmt='o', elinewidth=1, capthick=1, label=r'Minimum $\chi^2$')
plt.axvspan(optimal_A_V - A_V_uncertainty, optimal_A_V + A_V_uncertainty,
            alpha=0.3, color='gray', label=f'1σ uncertainty')
plt.xlabel('$A_{V}$ values')
plt.ylabel(r'$\chi^{2}$')
plt.legend()
plt.show()

# plot the distribution of perturbed A_V values
plt.hist(A_V_perturbed_values, bins=30, density=True, alpha=0.7, color='steelblue',
         edgecolor='black', label='MC Distribution')
plt.axvline(optimal_A_V, color='red', linestyle='--',
            label=f'Original fit: {optimal_A_V:.3f}')
plt.axvline(A_V_mean, color='green', linestyle=':',
            label=f'MC mean: {A_V_mean:.3f}')
plt.axvline(A_V_median, color='orange', linestyle=':',
            label=f'MC median: {A_V_median:.3f}')
plt.axvspan(optimal_A_V - A_V_std, optimal_A_V + A_V_std,
            alpha=0.2, color='gray', label=f'1σ std: ±{A_V_std:.3f}')
plt.xlabel('$A_{V}$')
plt.ylabel('Frequency')
plt.legend()
plt.show()
