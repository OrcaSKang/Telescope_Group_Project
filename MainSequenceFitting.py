"""
Author: Finlay Sime


"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from astropy.table import Table
import json
import astropy.units as u
import math as m

# data paths
ms_data_path = 'main_sequence.dat'
cluster_path = 'final_catalogue_M52.fits'
# Assuming data is in a structured format like CSV or similar
ms_data = pd.read_csv(ms_data_path, sep='\\s+', comment='#', engine='python')
cluster_data = Table.read(cluster_path, format='fits')

# convert to numpy array indexing
ms_data = ms_data.values

####################################
# cluster data is organised as:
# thresh npix tnpix ... mag_v magerr_v mab_b magerr_b mag_u magerr_u ...
# [0]    [1]  [2]   ...

# main sequence data is organised as:
# SpT Teff logT BCv Mv logL B-V
####################################

# extinction coefficient
A_V = 1.112
A_V_err = 0.0000775
R_V = 3.1
ExtBV = A_V / R_V
print(f"E(B-V) = {ExtBV}")

# Load data from Dust_Correction2.py
with open("MainSequenceFittingList.txt", "r") as file:
    MainSequenceFittingList = json.load(file)
print("Loaded list")

# extract data
cluster_mag_v = []
cluster_magerr_v = []
cluster_bv = []
cluster_err_bv = []
for i in range(0, len(MainSequenceFittingList)):
    cluster_mag_v.append(MainSequenceFittingList[i][0])
    cluster_magerr_v.append(MainSequenceFittingList[i][1])
    cluster_bv.append(MainSequenceFittingList[i][2])
    cluster_err_bv.append(MainSequenceFittingList[i][3])

cluster_mag_v = np.array(cluster_mag_v)
cluster_bv = np.array(cluster_bv)

cluster_mag_v_true = cluster_mag_v - A_V
cluster_bv_true = cluster_bv - ExtBV

delete_index = []
for i in range(len(cluster_bv_true)):
    if cluster_bv_true[i] < 0: # remove unphysical data point
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
chiSquared = []
for i in range(len(d)):
    dis = d[i]
    cluster_Mag_v = cluster_mag_v_true - 5*m.log10(dis/10)

    residual = cluster_Mag_v - f1(cluster_bv_true)

    chi = np.sum((residual/cluster_magerr_v)**2)
    chiSquared.append(chi)

min_index_chi = np.argmin(chiSquared) # minimise chi-squared
distance = d[min_index_chi] # find most likely distance to cluster via minimum chi-squared

cluster_Mag_v = cluster_mag_v_true - 5*m.log10(distance/10)
print(f"Distance to star cluster: {distance} pc")

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

plt.plot(d, chiSquared)
plt.xlabel('Distance (pc)')
plt.ylabel(r'$\chi^2$')
plt.show()
