import numpy as np
from astropy.io import fits
import os
import astroalign as aa

# Root directory
root_dir = ""

###################### MASTER FLATS & BIAS ################################

# Find the master bias and flat fits files
master_bias_path = os.path.join(root_dir, "master_bias.fits")
master_flat_B_path = os.path.join(root_dir, "master_flat_B.fits")
master_flat_U_path = os.path.join(root_dir, "master_flat_U.fits")
master_flat_V_path = os.path.join(root_dir, "master_flat_V.fits")

# Open maser bias and flat files for each filter
with fits.open(master_flat_B_path) as hdul:
    master_flat_B = hdul[0].data
with fits.open(master_flat_U_path) as hdul:
    master_flat_U = hdul[0].data
with fits.open(master_flat_V_path) as hdul:
    master_flat_V = hdul[0].data
with fits.open(master_bias_path) as hdul:
    master_bias = hdul[0].data

def stack(target_name):

    file_dir = os.path.join(root_dir, target_name)

    storage_V = np.zeros(shape=master_bias.shape)
    storage_U = np.zeros(shape=master_bias.shape)
    storage_B = np.zeros(shape=master_bias.shape)

    filters = {
    "Filter_U": (master_flat_U, storage_U, 60),
    "Filter_B": (master_flat_B, storage_B, 10),
    "Filter_V": (master_flat_V, storage_V, 10)
    }

    for filt, (master_flat, storage, expo_time) in filters.items():
        print(f"\n--- Processing {filt} images ---")

        reference_image = None
        header_ref = None
        count = 0
        
        # Making the reference image for the stack
        for file in os.listdir(file_dir):
            if filt in file and file.lower().endswith(".fits"):
                file_path = os.path.join(file_dir, file)
                with fits.open(file_path) as hdul:
                    data = hdul[0].data
                    header_ref = hdul[0].header
                
                reduced_data = (data - master_bias) / master_flat
                reference_image = (reduced_data)
                print(f"Reference image set for {filt}: {file}")
                break

        # Stacking
        for file in os.listdir(file_dir):
            if filt in file and file.lower().endswith(".fits"):
                file_path = os.path.join(file_dir, file)
                with fits.open(file_path) as hdul:
                    data = hdul[0].data
                    header = hdul[0].header
                
                reduced_data = (data - master_bias) / master_flat
                if np.array_equal(reduced_data, reference_image):
                    storage += reduced_data
                    count += 1
                    print(f"Stored {file}.")
                else:
                    try:
                        aligned_image, footprint = aa.register(reduced_data, reference_image)
                        print(f"Aligned {file} to {filt} reference image.")
                        storage += aligned_image
                        count += 1
                    except aa.MaxIterError:
                        print(f"Alignment failed for {file}: max iterations reached")
                        continue
                    
                        
        # Save the final image file
        final_image = storage / (expo_time * count)
        safe_name_ini = target_name.replace("Standard/", "")
        safe_name = safe_name_ini.replace("/", "_")
        aligned_filename = f"Final_{safe_name}_{filt}.fits"
        aligned_path = os.path.join(root_dir, aligned_filename)
        fits.writeto(aligned_path, final_image, header_ref, overwrite=True)

    print(f"finished for {target_name}")


def main():

    stack("M52")

    stack("NGC6755")

    stack("Standard/11223/Beginning")

    stack("Standard/11223/End")
    
    stack("Standard/111773/Beginning")

    stack("Standard/114750/Beginning")
    
    stack("Standard/114750/End")

    stack("Standard/GD246/Beginning")

    stack("Standard/PG2317/Beginning")

    stack("Standard/PG2317/End")

    print("\n---FINISHED---")

# The targets that have both the first and last observation files are 11223, 114750, and PG2317.

if __name__ == "__main__":
    main()