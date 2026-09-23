# EMP5027-rows-to-pixels
Python for environmental data science. EMP5027, MSc Earth Science, University of Malta. From core Python and pandas through geospatial modelling to deep learning on satellite, plant-disease, and field imagery (CNNs, transfer learning, Grad-CAM). Runs on Colab.

# rows-to-pixels

Course notebooks for **EMP5027: Methods in Data Analysis & Quality Assurance**, MSc Earth Science, University of Malta.

The module teaches Python for environmental data science end to end: starting from core language and data-handling skills, through statistical analysis and geospatial modelling, and finishing with deep learning on environmental imagery. The dataset used most often along the way is Palmer Penguins, hence *rows* (tabular data, where the course starts) *to pixels* (image data, where it ends up).

## Contents

| # | Topic | Status |
|---|---|---|
| 1 | Python foundations for data analysis | to be added |
| 2 | Data structures, functions, and modules | to be added |
| 3 | Working with structured data (pandas) | to be added |
| 4 | Exploratory data analysis & statistics | to be added |
| 5 | Geospatial data for environmental science | to be added |
| 6 | **Deep learning for environmental imagery** | **available now** |

Practicals are being uploaded incrementally; deep learning (Practical 6) is complete and the current focus of this repository.

## Practical 6: Deep Learning

Three self-contained notebooks, each applying the same core toolkit, including a CNN trained from scratch, transfer learning with a pretrained backbone, and Grad-CAM for interpretability. This will apply to different environmental imaging problems. The notebooks are meant to be worked through in order, but each stands alone.

| Notebook | Application | Dataset | Open in Colab |
|---|---|---|---|
| **6a** | Satellite land-cover classification | [EuroSAT](https://github.com/phelber/EuroSAT). Sentinel-2 patches, 10 land-use classes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/rows-to-pixels/blob/main/notebooks/practical-6-deep-learning/EMP5027-Lecture-6a-Satellite-Land-Cover-Classification-EuroSAT.ipynb) | 
| **6b** | Plant disease diagnosis | [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset). Leaf photos, healthy vs. diseased | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/rows-to-pixels/blob/main/notebooks/practical-6-deep-learning/EMP5027-Lecture-6b-Plant-Disease-Diagnosis-PlantVillage.ipynb) |
| **6c** | Invasive weed species identification | [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds). Real field photos, imbalanced classes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbutler2405/rows-to-pixels/blob/main/notebooks/practical-6-deep-learning/EMP5027-Lecture-6c-Weed-Species-Identification-DeepWeeds.ipynb) |

Each notebook works through the same progression:

1. **Load and explore** the dataset via `tensorflow_datasets`. No manual download or Kaggle account required.
2. **Build a `tf.data` pipeline**: resizing, normalisation, augmentation, batching.
3. **Train a CNN from scratch**, to see what a network learns with no prior knowledge.
4. **Apply transfer learning**, fine-tuning a pretrained ImageNet backbone (MobileNetV2), usually the biggest single improvement on a modest dataset.
5. **Evaluate properly**: classification report and confusion matrix, not accuracy alone.
6. **Explain predictions with Grad-CAM**, checking whether the model is actually looking at the relevant part of the image.

6c additionally covers **class imbalance**. Real field data rarely arrives balanced, so this notebook handles it directly with class weighting and balanced accuracy, rather than assuming it away.

## Running the notebooks

**Recommended: Google Colab.** Click any badge above. TensorFlow is preinstalled on Colab; each notebook's setup cell installs the one extra package it needs (`tensorflow_datasets`). For a large speed-up, switch to a GPU runtime first: `Runtime >> Change runtime type >> Hardware accelerator >> GPU`.

**Running locally** works too, provided the packages in `requirements.txt` are installed:

```bash
pip install -r requirements.txt
jupyter notebook
```

Colab sessions are temporary. If you want to keep a trained model or exported results, mount Google Drive or download the file before the session ends (both are covered in each notebook's setup cell).

## Repository structure

```
rows-to-pixels/
├── README.md
├── requirements.txt
└── notebooks/
    └── practical-6-deep-learning/
        ├── EMP5027-Lecture-6a-Satellite-Land-Cover-Classification-EuroSAT.ipynb
        ├── EMP5027-Lecture-6b-Plant-Disease-Diagnosis-PlantVillage.ipynb
        └── EMP5027-Lecture-6c-Weed-Species-Identification-DeepWeeds.ipynb
```

## Author

Dr. Liam Butler 
Department of Systems & Control Engineering/Institute of Earth Systems
University of Malta.
