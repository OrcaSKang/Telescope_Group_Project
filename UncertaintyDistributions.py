from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt

cluster_path = 'final_catalogue_M52.fits'

cluster_data = Table.read(cluster_path, format='fits')
#print(*cluster_data)

cluster_mag_v = np.array(cluster_data['mag_v'].tolist())
cluster_mag_b = np.array(cluster_data['mag_b'].tolist())
cluster_mag_u = np.array(cluster_data['mag_u'].tolist())
cluster_magerr_v = np.array(cluster_data['magerr_v'].tolist())
cluster_magerr_b = np.array(cluster_data['magerr_b'].tolist())
cluster_magerr_u = np.array(cluster_data['magerr_u'].tolist())

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

counts, bins = np.histogram(cluster_err_bv, bins=np.linspace(min(cluster_err_ub), max(cluster_err_ub), 1000))

mean = np.mean(cluster_err_ub)
print(mean)

remove_index = []
for i in range(0, len(cluster_err_ub)):
    if cluster_err_ub[i] >= 0.05:
        remove_index.append(i)
    elif cluster_err_bv[i] >= 0.05:
        remove_index.append(i)

j = np.array(remove_index)
cluster_err_ub = np.delete(cluster_err_ub, j)
cluster_err_bv = np.delete(cluster_err_bv, j)

fig, ax = plt.subplot_mosaic([['UB', 'BV']], figsize=(14, 5))

ax['UB'].hist(abs(cluster_err_ub), bins=100)
ax['UB'].set_xlabel(r'$\sigma$(U-B)')
ax['UB'].set_ylabel('No. in uncertainty bin')

ax['BV'].hist(abs(cluster_err_bv), bins=100)
ax['BV'].set_xlabel(r'$\sigma$(B-V)')
ax['BV'].set_ylabel('No. in uncertainty bin')

plt.show()
