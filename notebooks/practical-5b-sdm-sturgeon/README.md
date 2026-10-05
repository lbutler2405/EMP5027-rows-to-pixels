# Practical 5b: Species Distribution Modelling (Atlantic Sturgeon, April)

A full habitat suitability workflow: presence points, buffer-based pseudo-absences, an aligned stack of seven environmental rasters, four classifiers (logistic regression, random forest, SVM, MLP), an AUC-weighted ensemble mapped to GeoTIFF, and an explainable AI section (built-in and permutation importance, partial dependence plots, SHAP).

## 1. Get the data

The environmental rasters, presence points and study extent are bundled in **[Practical5_SDM_data.zip](Practical5_SDM_data.zip)** (about 17 MB), which sits in this folder. Download it (click the link, then the download button on the file page) and unzip it **into this folder**, so that the notebook sits next to three new folders:

```
practical-5b-sdm-sturgeon/
├── EMP5027-Practical-5b-SDM-Geospatial-Sturgeon.ipynb
├── presence/       (sturgeon presence points, shapefile)
├── extent/         (study area, shapefile)
└── rasters/
    ├── depth/
    ├── substrate/
    ├── photoperiod/
    ├── precipitation/
    ├── sst/
    └── sss_climatology_by_month/Apr/
```

The notebook reads these with relative paths, so the folders must be exactly here. Unzipping creates them for you.

## 2. Install the packages

This practical runs locally rather than in Colab. From the repository root:

```bash
conda create -n emp5027 python=3.11 -y
conda activate emp5027
pip install -r requirements.txt
```

If a geospatial package fails to install with `pip` (this sometimes happens with `rasterio` or `geopandas`), install the geospatial stack from conda-forge instead:

```bash
conda install -c conda-forge geopandas rasterio rioxarray contextily
```

## 3. Run

Open the notebook in JupyterLab and run it from the top. A full run takes a few minutes on a laptop. The notebook writes prediction maps (GeoTIFF files) next to itself, and these are ignored by git.
