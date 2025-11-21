from astropy.io import fits
import astroalign as aa
from pathlib import Path
import numpy as np


root_dir = Path("")

aligned_dir = root_dir / "Aligned_Files"
aligned_dir.mkdir(exist_ok=True)

def align(target_name):
    files = sorted(root_dir.glob(f"*{target_name}*.fits"))
    if not files:
        print(f"No fits files found for target '{target_name}'")
        return

    reference_file = None
    reference_image = None
    header_ref = None

    for p in files:
        if "V" in p.name:
            reference_file = p.name
            with fits.open(p) as hdul:
                reference_image = hdul[0].data.astype(np.float32)
            print(f"Reference image selected for {target_name}: {p.name}")
            break

    if reference_image is None:
        print(f"No reference (V) file found for target '{target_name}'")
        return

    for p in files:
        if p.name == reference_file:
            with fits.open(p) as hdul:
                data = hdul[0].data.astype(np.float32)
                header = hdul[0].header
            ref_path = aligned_dir / p.name
            fits.writeto(ref_path, data, header, overwrite=True)
            print(f"Saved {p.name} → {ref_path}")
            continue

        with fits.open(p) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header

        try:
            aligned, footprint = aa.register(data, reference_image, max_control_points=250, detection_sigma=2.0)
            aligned_path = aligned_dir / p.name
            fits.writeto(aligned_path, aligned, header, overwrite=True)
            print(f"Aligned {p.name} → {aligned_path}")
        except Exception as e:
            print(f"Failed to align {p.name}: {e}")


def main():
    align("M52")
    align("NGC6755")
    align("11223_Beginning")
    align("11223_End")              # B is the reference
    align("111773_Beginning")       # U astroalign failed
    align("114750_Beginning")
    align("114750_End")
    align("GD246_Beginning")
    align("PG2317_Beginning")
    align("PG2317_End")


if __name__ == "__main__":
    main()
