import os
from astropy.io import fits

# Root directory
root_dir = ""

# Trimmed file saving folder
output_root = os.path.join(root_dir, "Trimmed_files")

# Loop through all files in the folder (including subfolders)
for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.lower().endswith(".fits"):
            file_path = os.path.join(dirpath, file)
            print(f"Processing: {file_path}")

            # Open FITS file
            with fits.open(file_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            # Values are taken from trimming.py
            ymin, ymax = 42, 4053
            xmin, xmax = 42, 4053

            # Trim data
            trimmed_data = data[ymin:ymax+1, xmin:xmax+1]

            print(data.shape)
            print(trimmed_data.shape)

            # Create output file path
            rel_path = os.path.relpath(dirpath, root_dir)
            save_dir = os.path.join(output_root, rel_path)

            # Create the sub-folders
            # os.makedirs(save_dir, exist_ok=True)

            out_file = os.path.join(save_dir, file)

            # Save new FITS
            # hdu = fits.PrimaryHDU(trimmed_data, header=header)
            # hdu.writeto(out_file, overwrite=True)

            print(f"Saved: {out_file}")