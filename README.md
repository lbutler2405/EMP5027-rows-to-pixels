# EMP5027-rows-to-pixels

Course notebooks for **EMP5027, Methods in Data Analysis and Quality Assurance**, MSc Earth Science, University of Malta.

The module teaches Python for environmental data science end to end, starting from core language and data-handling skills, through statistical analysis and geospatial modelling, and finishing with deep learning on environmental imagery. The dataset used most often along the way is Palmer Penguins, hence *rows* (tabular data, where the course starts) *to pixels* (image data, where it ends up).

All notebooks open directly in Google Colab via the badges below, no local setup required. Each one includes a short Colab setup cell that installs anything not preinstalled there and, where a notebook needs its own CSV data, fetches that data automatically from this repository the first time it is needed.

## Contents

| # | Notebook | Topic | Open in Colab |
|---|---|---|---|
| 1 | Intro to Python | Python foundations for data analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-1-intro-to-python/Practical_1_Intro_to_Python.ipynb) |
| 2 | Lecture 2 | Data structures, functions, and modules | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-2-data-structures-functions-modules/EMP5027-Lecture-2-Data-Structures-Functions-Modules.ipynb) |
| 2b | Occam's Razor | Model simplicity vs. overfitting (linear, polynomial, random forest) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-2b-occams-razor/EMP5027-Practical-Occams-Razor.ipynb) |
| 3 | Lecture 3 | Working with structured data (pandas deep dive and PCA) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-3-structured-data/EMP5027-Lecture-3-Working-with-Structured-Data.ipynb) |
| 3b | Lecture 3b | Time-aware data, multi-station regression, joins | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-3b-time-aware-data-joins/EMP5027-Lecture-3b-Time-Aware-Data-and-Joins.ipynb) |
| 4 | Lecture 4 | Exploratory data analysis and statistics (incl. GLM/GAM, clustering, spatial) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-4-eda-stats/EMP5027-Lecture-4-EDA-and-Stats.ipynb) |
| 5 | Lecture 5 | Geospatial data fundamentals for environmental science | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-5-geospatial-fundamentals/EMP5027-Lecture-5-Geospatial-Data-for-Environmental-Science.ipynb) |
| 6 | **Deep learning** | **CNNs, transfer learning, Grad-CAM on environmental imagery** | see below |

## Practical 6: Deep Learning

Three self-contained notebooks, each applying the same core toolkit, a CNN trained from scratch, transfer learning with a pretrained backbone, and Grad-CAM for interpretability, to a different environmental imaging problem. They are meant to be worked through in order, but each stands alone.

| Notebook | Application | Dataset | Open in Colab |
|---|---|---|---|
| **6a** | Satellite land-cover classification | [EuroSAT](https://github.com/phelber/EuroSAT), Sentinel-2 patches, 10 land-use classes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-6-deep-learning/EMP5027-Lecture-6a-Satellite-Land-Cover-Classification-EuroSAT.ipynb) |
| **6b** | Plant disease diagnosis | [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset), leaf photos, healthy vs. diseased | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-6-deep-learning/EMP5027-Lecture-6b-Plant-Disease-Diagnosis-PlantVillage.ipynb) |
| **6c** | Invasive weed species identification | [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds), real field photos, imbalanced classes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-6-deep-learning/EMP5027-Lecture-6c-Weed-Species-Identification-DeepWeeds.ipynb) |

Each notebook works through the same progression:

1. **Load and explore** the dataset via `tensorflow_datasets`, no manual download or Kaggle account required.
2. **Build a `tf.data` pipeline**: resizing, normalisation, augmentation, batching.
3. **Train a CNN from scratch**, to see what a network learns with no prior knowledge.
4. **Apply transfer learning**, fine-tuning a pretrained ImageNet backbone (MobileNetV2), usually the biggest single improvement on a modest dataset.
5. **Evaluate properly**: classification report and confusion matrix, not accuracy alone.
6. **Explain predictions with Grad-CAM**, checking whether the model is actually looking at the relevant part of the image.

6c additionally covers **class imbalance**: real field data rarely arrives balanced, so this notebook handles it directly with class weighting and balanced accuracy, rather than assuming it away.

## Data

Most notebooks either generate their own small example datasets in the first cell or use datasets that ship with the libraries themselves (seaborn's penguins, sklearn's iris, `tensorflow_datasets` for the deep learning notebooks), so there is nothing to download.

Lecture 3b and Lecture 4 are the exception: they use real CSV files that are bundled in this repository alongside their notebooks. When one of these two is opened via its Colab badge, its setup cell checks whether the required CSVs are already present locally and, if not, downloads them automatically from this repository before the notebook needs them. Nothing to do on your end, just run the cells in order.

## Not yet in this repository

The applied **species distribution modelling** practical (raster-based habitat suitability modelling for Atlantic Sturgeon, using logistic regression, random forest, SVM, MLP, and an ensemble) depends on several hundred MB of raster data, which does not belong in a git repository. It runs locally, alongside its own data folder, rather than through Colab.

## Running the notebooks

**Recommended: Google Colab.** Click any badge above. Each notebook's setup cell installs whatever it needs beyond Colab's defaults, for example `geopandas`, `contextily`, and `pygam` for the geospatial and statistics notebooks, `tensorflow_datasets` for the deep learning ones. For the deep learning notebooks especially, switch to a GPU runtime first for a large speed-up: `Runtime > Change runtime type > Hardware accelerator > GPU`.

**Running locally** works too, provided the packages in `requirements.txt` are installed:

```bash
pip install -r requirements.txt
jupyter notebook
```

Colab sessions are temporary. If you want to keep a trained model or exported results, mount Google Drive or download the file before the session ends (both are covered in each notebook's setup cell).

## Repository structure

```
EMP5027-rows-to-pixels/
├── README.md
├── requirements.txt
└── notebooks/
    ├── practical-1-intro-to-python/
    ├── practical-2-data-structures-functions-modules/
    ├── practical-2b-occams-razor/
    ├── practical-3-structured-data/
    ├── practical-3b-time-aware-data-joins/      (+ Parts3-4/ data)
    ├── practical-4-eda-stats/                   (+ CSV data)
    ├── practical-5-geospatial-fundamentals/
    └── practical-6-deep-learning/
```

## Author

Dr. Liam Butler, Department of Systems and Control Engineering / Institute of Earth Systems, University of Malta.
