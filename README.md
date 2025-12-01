# Telescope_Group_Project

This repository contains the code developed as part of the Telescope Group Project undertaken during the 2025/26 academic year.

The aim of the project was to construct cluster catalogues, determine membership, and produce Hertzsprung–Russell (HR) diagrams. 
HR diagrams of a young cluster and an old cluster were compared to investigate differences in their stellar populations and evolutionary stages.

The analysis required a complete workflow spanning image reduction, alignment, photometry, catalogue construction, and astrophysical interpretation.

Two open clusters were analysed: a young cluster (NGC 6754, also known as M52) and an old cluster (NGC 6755).

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

### 3. Cluster Catalogue Construction
Source extraction was performed using `sep`, producing catalogues for two observed clusters:

- Background estimation  
- Object detection and photometry  
- Catalogue construction for each filter  
- Transfer to TOPCAT via SAMP for inspection and cross-matching  

These catalogues provide the data points required to construct the HR diagram.

---

## Dependencies

This project uses Python and the following key packages:

- `astroalign` — for image alignment  
- `sep` — for source extraction and background estimation 
- `numpy`, `scipy`, `matplotlib` — general scientific computing tools  
- `astropy` — for FITS handling and astronomical utilities  

It will be kept updated as the project progresses