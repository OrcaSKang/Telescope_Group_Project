import pandas as pd
import matplotlib.pyplot as plt
from isochrones import get_ichrone
from astropy.table import Table
import numpy as np
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')

M52_data_path = 'final_catalogue_M52_2sigma.fits'
cluster_data = Table.read(M52_data_path, format='fits')

snr_threshold = 3
ok = ((cluster_data['flux_v']/cluster_data['fluxerr_v'] > snr_threshold) &
      (cluster_data['flux_b']/cluster_data['fluxerr_b'] > snr_threshold))
cluster_data = cluster_data[ok]

cluster_mag_v = np.array(cluster_data['mag_v'], dtype=float)
cluster_mag_b = np.array(cluster_data['mag_b'], dtype=float)
cluster_bv = cluster_mag_b - cluster_mag_v

RV = 3.1
AV = 3.144
EBV = AV / RV
dis = 1315

cluster_mag_v_true = cluster_mag_v - AV
cluster_bv_true = cluster_bv - EBV
cluster_Mag_v = cluster_mag_v_true - 5 * np.log10(dis / 10)

mask = cluster_bv_true >= -0.32
c_bv = cluster_bv_true[mask]
c_mag = cluster_Mag_v[mask]

iso = get_ichrone('mist', bands=['B', 'V'])

age_values = np.arange(7.5, 8.5, 0.02)
feh_values = np.arange(-0.5, 0.5, 0.05)

all_results = [] 

print("Starting fitting...")


for current_age in age_values:
    for current_feh in feh_values:
        try:
            model_df = iso.isochrone(age=current_age, feh=current_feh)
            m_bv = model_df['B_mag'] - model_df['V_mag']
            m_mag = model_df['V_mag']
            
            model_points = np.column_stack([m_bv, m_mag])
            obs_points = np.column_stack([c_bv, c_mag])
            
            tree = cKDTree(model_points)
            distances, _ = tree.query(obs_points)
            current_chi2 = np.sum(distances**2)
            
            all_results.append({'age': current_age, 'feh': current_feh, 'chi2': current_chi2})
        except:
            continue


res_df = pd.DataFrame(all_results)

min_chi2 = res_df['chi2'].min()
best_fit = res_df.loc[res_df['chi2'].idxmin()]

threshold = min_chi2 * 1.10 
good_fits = res_df[res_df['chi2'] <= threshold]

age_best, age_err = best_fit['age'], good_fits['age'].std()
feh_best, feh_err = best_fit['feh'], good_fits['feh'].std()

print(f"\n" + "="*30)
print(f"Final Fitting Results:")
print(f"Age (logAge): {age_best:.3f} ± {age_err:.3f}")
print(f"Metallicity ([Fe/H]): {feh_best:.3f} ± {feh_err:.3f}")
print(f"Age in Myr: {10**age_best/1e6:.2f} Myr")
print(f"="*30)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

ax1.scatter(c_bv, c_mag, s=2, c='gray', alpha=0.3, label='M52 Data')
best_model_df = iso.isochrone(age=age_best, feh=feh_best)
m_bv_best = best_model_df['B_mag'] - best_model_df['V_mag']
m_mag_best = best_model_df['V_mag']

ax1.plot(m_bv_best, m_mag_best, 'r-', lw=2, label=f'Best-fit ({10**age_best/1e6:.1f} Myr)')
ax1.invert_yaxis()
ax1.set_xlabel('(B-V) corrected')
ax1.set_ylabel('V (Absolute Magnitude)')
ax1.set_title('Color-Magnitude Diagram Fitting')
ax1.legend()

pivot = res_df.pivot(index='age', columns='feh', values='chi2')
im = ax2.imshow(pivot, extent=[feh_values.min(), feh_values.max(), age_values.min(), age_values.max()],
               aspect='auto', origin='lower', cmap='viridis_r')
ax2.plot(feh_best, age_best, 'rx', markersize=15, markeredgewidth=3, label='Best-fit point')
fig.colorbar(im, ax=ax2, label='Chi-square value (Lower is better)')
ax2.set_xlabel('[Fe/H]')
ax2.set_ylabel('log(Age)')
ax2.set_title('Confidence Map (Error Estimation)')
ax2.legend()

plt.tight_layout()
plt.show()