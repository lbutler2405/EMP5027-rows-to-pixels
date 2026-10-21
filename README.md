# rows-to-pixels

Course notebooks for **EMP5027 Methods in Data Analysis & Quality Assurance**, MSc Earth Science, University of Malta.

The module teaches Python for environmental data science end to end, from core language and data-handling skills, through statistical analysis and geospatial modelling, to deep learning on environmental imagery. The dataset used most often along the way is Palmer Penguins, hence *rows* (tabular data, where the course starts) *to pixels* (image data, where it ends up).

## Contents

| # | Notebook | Topic | Open in Colab |
|---|---|---|---|
| 1 | Intro to Python | Python foundations for data analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-1-intro-to-python/Practical_1_Intro_to_Python.ipynb) |
| 2 | Data structures, functions and modules | Lists, dictionaries, NumPy, pandas basics, functions, modules | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-2-data-structures-functions-modules/EMP5027-Lecture-2-Data-Structures-Functions-Modules.ipynb) |
| 2b | Occam's Razor | Model simplicity versus overfitting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-2b-occams-razor/EMP5027-Practical-Occams-Razor.ipynb) |
| 3 | Working with structured data | A pandas deep-dive | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-3-structured-data/EMP5027-Lecture-3-Working-with-Structured-Data.ipynb) |
| 3b | Time-aware data and joins | Time series, multi-station data, joins | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-3b-time-aware-data-joins/EMP5027-Lecture-3b-Time-Aware-Data-and-Joins.ipynb) |
| 4 | Exploratory data analysis and statistics | EDA, statistical testing and modelling, spatial plots | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-4-eda-stats/EMP5027-Lecture-4-EDA-and-Stats.ipynb) |
| 5 | Geospatial data fundamentals | Vector and raster data for environmental science | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-5-geospatial-fundamentals/EMP5027-Lecture-5-Geospatial-Data-for-Environmental-Science.ipynb) |
| 6 | Deep learning | CNNs, transfer learning and Grad-CAM on environmental imagery | see below |

## Practical 5b: Species Distribution Modelling

This practical builds a full habitat suitability workflow for the Atlantic sturgeon: presence points, pseudo-absences, an aligned stack of environmental rasters, four classifiers (logistic regression, random forest, SVM and a small neural network), an AUC-weighted ensemble, and a section on **explainable AI** (permutation importance, partial dependence plots and SHAP).

The data (rasters, presence points and study extent) comes as a single small zip file in that practical's folder. See the [practical's README](notebooks/practical-5b-sdm-sturgeon/README.md) for where to unzip it. This notebook is designed to run locally rather than in Colab.

## Practical 6: Deep Learning

Three standalone image-classification notebooks, each applying the same toolkit (a CNN trained from scratch, transfer learning with a pretrained backbone, and Grad-CAM for interpretability) to a different environmental imaging problem.

| Notebook | Application | Dataset | Open in Colab |
|---|---|---|---|
| **6a** | Satellite land-cover classification | [EuroSAT](https://github.com/phelber/EuroSAT), Sentinel-2 patches, 10 land-use classes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-6-deep-learning/EMP5027-Lecture-6a-Satellite-Land-Cover-Classification-EuroSAT.ipynb) |
| **6b** | Plant disease diagnosis | [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset), leaf photos, healthy versus diseased | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-6-deep-learning/EMP5027-Lecture-6b-Plant-Disease-Diagnosis-PlantVillage.ipynb) |
| **6c** | Invasive weed species identification | [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds), real field photos, imbalanced classes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/EMP5027-rows-to-pixels/blob/main/notebooks/practical-6-deep-learning/EMP5027-Lecture-6c-Weed-Species-Identification-DeepWeeds.ipynb) |

Each notebook downloads its dataset automatically with `tensorflow_datasets`, so no manual download or Kaggle account is needed.

**Use Google Colab for this practical.** Click a badge, then choose **Runtime > Change runtime type > GPU** before running anything. Colab already has TensorFlow installed and gives you a free GPU, which makes training much faster.

**Running on your own laptop instead?** It works, but on the CPU, and training will be slower. Run the [Practical 6 setup notebook](notebooks/practical-6-deep-learning/EMP5027-Lecture-6-Setup-Run-First-if-Not-Using-Colab.ipynb) first. It installs TensorFlow, checks whether a GPU is available, and pre-downloads the datasets. TensorFlow has no official GPU support on macOS, and no native Windows GPU support beyond version 2.10, so most laptops will use the CPU.

## Running the notebooks

**Recommended: Google Colab** for practicals 1 to 6. Click any badge above. Each notebook's setup cell installs whatever it needs beyond Colab's defaults, and notebooks that need data files download them from this repository automatically.

**Running locally** also works, provided the packages in `requirements.txt` are installed. The [Anaconda setup guide](docs/EMP5027_Anaconda_Setup_and_Packages.pdf) walks through the installation. A fresh conda environment avoids most version conflicts:

```bash
conda create -n emp5027 python=3.11 -y
conda activate emp5027
pip install -r requirements.txt
jupyter lab
```

Colab sessions are temporary. If you want to keep a trained model or exported results, mount Google Drive or download the file before the session ends.

## Repository structure

```
EMP5027-rows-to-pixels/
├── README.md
├── requirements.txt
├── docs/
│   └── EMP5027_Anaconda_Setup_and_Packages.pdf
└── notebooks/
    ├── practical-1-intro-to-python/
    ├── practical-2-data-structures-functions-modules/
    ├── practical-2b-occams-razor/
    ├── practical-3-structured-data/
    ├── practical-3b-time-aware-data-joins/      (+ Parts3-4/ data files)
    ├── practical-4-eda-stats/                    (+ CSV data files)
    ├── practical-5-geospatial-fundamentals/
    ├── practical-5b-sdm-sturgeon/                (+ data zip, unzip before running)
    └── practical-6-deep-learning/
```

The data files in the practical 3b and practical 4 folders are downloaded by those notebooks directly from this repository, so please keep the folder names and file names as they are.

## Author

Dr Liam Butler, Department of Systems & Control Engineering, University of Malta.
