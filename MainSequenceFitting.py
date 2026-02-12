import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from astropy.table import Table
import math as m

# data paths
ms_data_path = 'main_sequence.dat'
cluster_path = 'final_catalogue_M52.fits'
cluster_path_2sigma = 'final_catalogue_M52_2sigma.fits'
cluster_path_NGC6755 = 'final_catalogue_NGC6755_2sigma.fits'
# Assuming data is in a structured format like CSV or similar
ms_data = pd.read_csv(ms_data_path, sep='\\s+', comment='#', engine='python')
cluster_data = Table.read(cluster_path, format='fits')

# convert to numpy array indexing
ms_data = ms_data.values

# filter out points beneath a SNR threshold
snr_threshold = 3
ok = ((cluster_data['flux_v']/cluster_data['fluxerr_v'] > snr_threshold) &
      (cluster_data['flux_b']/cluster_data['fluxerr_b'] > snr_threshold))
cluster_data = cluster_data[ok]

cluster_mag_v = np.array(cluster_data['mag_v'].tolist())
cluster_mag_b = np.array(cluster_data['mag_b'].tolist())
cluster_magerr_v = np.array(cluster_data['magerr_v'].tolist())
cluster_magerr_b = np.array(cluster_data['magerr_b'].tolist())
cluster_mag_v = cluster_mag_v.astype(float)
cluster_mag_b = cluster_mag_b.astype(float)
cluster_magerr_v = cluster_magerr_v.astype(float)
cluster_magerr_b = cluster_magerr_b.astype(float)
# get colours
cluster_bv = cluster_mag_b - cluster_mag_v
cluster_err_bv = np.sqrt(cluster_magerr_v**2 + cluster_magerr_b**2)

# extinction coefficient
A_V = 3.126 # M52
#A_V = 2.151 # NGC 6755
A_V_err = 0.632 # M52
#A_V_err = 1.294 # NGC 6755
R_V = 3.1
ExtBV = A_V / R_V
print(f"E(B-V) = {ExtBV}")

cluster_mag_v = np.array(cluster_mag_v)
cluster_bv = np.array(cluster_bv)

cluster_mag_v_true = cluster_mag_v - A_V
cluster_bv_true = cluster_bv - ExtBV

delete_index = []
for i in range(len(cluster_bv_true)):
    if cluster_bv_true[i] < 0: # remove data point causing error
        delete_index.append(i)

cluster_mag_v_true = np.delete(cluster_mag_v_true, delete_index)
cluster_mag_v = np.delete(cluster_mag_v, delete_index)
cluster_bv = np.delete(cluster_bv, delete_index)
cluster_magerr_v = np.delete(cluster_magerr_v, delete_index)
cluster_bv_true = np.delete(cluster_bv_true, delete_index)
cluster_err_bv = np.delete(cluster_err_bv, delete_index)

print(f"No. of stars: {len(cluster_bv_true)}") # check number of stars

Mv = ms_data[:,4] # Main sequence V-band magnitude
MSBV = ms_data[:,6] # Main sequence (B-V) colour

# sort data
Mv_MSBV = list(set(zip(Mv, MSBV)))
Mv_MSBV.sort(key=lambda x: x[1])
Mv = np.array([item[0] for item in Mv_MSBV])
MSBV = np.array([item[1] for item in Mv_MSBV])

# interpolating function
f1 = interpolate.interp1d(MSBV, Mv, kind='linear', bounds_error=False, fill_value='extrapolate')
x_min, x_max = min(MSBV), max(MSBV)
x = np.arange(x_min, x_max, 0.0001)
y = f1(x)

d = np.arange(1, 2000, 1) # create range of possible distances in parsecs

# calculate chi-squared value for each data point

# function to calculate chi-sqaured for A_V values
def ComputeChiSquared(d, v_data, bv_data, function):
    chiSquared = []

    for i in range(len(d)):
        dis = d[i]

        cluster_Mag_v = v_data - 5*m.log10(dis/10)
        residual = cluster_Mag_v - function(bv_data)

        expected_v = function(bv_data)

        # calculate chi-squared
        chi = np.sum((residual**2)/(expected_v+1e-20))
        chiSquared.append(chi)

    return chiSquared

chiSquaredOriginal = ComputeChiSquared(d, cluster_mag_v_true, cluster_bv_true, f1)

min_index_chi = np.argmin(chiSquaredOriginal) # minimise chi-squared
distance = d[min_index_chi] # find most likely distance to cluster via minimum chi-squared

distance_perturbed = []
M = 50
for iteration in range(M):
    print(f"    Iteration {iteration}/{M}")

    A_V_perturbed = np.random.normal(A_V, A_V_err)
    perturbed_mag_v = np.random.normal(cluster_mag_v, cluster_magerr_v)
    perturbed_bv = np.random.normal(cluster_bv, cluster_err_bv)

    perturbed_mag_v_true = cluster_mag_v - A_V_perturbed
    perturbed_bv_true = cluster_bv - (A_V_perturbed/R_V)

    chiSquaredPerturbed = ComputeChiSquared(d, perturbed_mag_v_true, perturbed_bv_true, f1)
    min_index_perturbed = np.argmin(chiSquaredPerturbed)
    distance_perturbed.append(d[min_index_perturbed])

distance_perturbed = np.array(distance_perturbed)
distance_mean = np.mean(distance_perturbed)
distance_std = np.std(distance_perturbed)
distance_median = np.median(distance_perturbed)

percentiles = np.percentile(distance_perturbed, [16, 50, 84])
lower_uncertainty = percentiles[1] - percentiles[0]
upper_uncertainty = percentiles[2] - percentiles[1]

print("Monte Carlo Results")
print(f"Optimal distance from original data: {distance:.4f}")
print(f"Mean distance from MC: {distance_mean:.4f}")
print(f"Median distance from MC: {distance_median:.4f}")
print(f"Standard deviation: {distance_std:.4}")

distance_uncertainty = distance_std

cluster_Mag_v = cluster_mag_v_true - 5*m.log10(distance/10)
print(f"Distance to star cluster: {distance} ± {distance_uncertainty} pc")

#plot data
plt.scatter(MSBV, Mv, s=2)
plt.plot(x, y, label='Absolute Data')
plt.scatter(cluster_bv, cluster_mag_v, s=2, label='Raw Data')
plt.scatter(cluster_bv_true, cluster_mag_v_true, s=2, label='Dust-corrected Data')
plt.scatter(cluster_bv_true, cluster_Mag_v, s=2, label='Distance-corrected Data')
plt.gca().invert_yaxis()
plt.ylabel(r'$M_V$')
plt.xlabel('(B-V)')
plt.legend()
plt.show()

plt.plot(d, chiSquaredOriginal)
plt.xlabel('Distance (pc)')
plt.ylabel(r'$\chi^2$')
plt.show()
