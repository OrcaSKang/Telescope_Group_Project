# Telescope_Group_Project

This repository contains the code developed as part of the Telescope Group Project undertaken during the 2025/26 academic year.  
The broader aim of the project was to construct a cluster catalogue, determine membership, and produce a Hertzsprung–Russell (HR) diagram for the selected cluster. The analysis required a complete workflow spanning image reduction, alignment, photometry, catalogue building, and astrophysical interpretation.

Only the initial stages of the workflow are included in this repository. These form the essential foundation on which the later analysis was built.

---

The scripts in this repository focus on:

### **1. CCD Image Reduction**
A full reduction pipeline was implemented to prepare raw telescope frames for scientific use.  
This includes:
- Bias subtraction  
- Flat-field correction  
- Dark correction (if applicable)  
- Cosmic-ray removal  
- Preparation of science-ready frames  

These steps ensure that instrumental and detector effects are removed before any further processing.

### **2. Cross-Filter Image Alignment (`astroalign`)**
Images taken in different filters were aligned using the `astroalign` package.  
This was necessary to:
- Match sources consistently across filters  
- Enable reliable colour measurements  
- Prepare for combined catalogues and HR diagram construction  

Accurate alignment was crucial for the later stages of the project, such as photometry, cluster membership analysis, and catalogue assembly.

---

## Dependencies

This project uses Python and the following key packages:

- `astroalign` — for image alignment  
- `sep` — for source extraction and background estimation 
- `numpy`, `scipy`, `matplotlib` — general scientific computing tools  
- `astropy` — for FITS handling and astronomical utilities  

It will be kept updated as the project progresses