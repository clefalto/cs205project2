## CS205 Project 2

Authors:
- Tandis Salem
- Garett Hammerle

### Repository structure:
- nearest_neighbor/
- feature_selection/
- dendrogram/

### Requirements:
numpy
pandas
matplotlib
scipy
ucimlrepo

### How to Run

#### Task 1: Feature Selection on Synthetic Datasets

```bash
python p2_t1_1nn.py
```

Implements a 1-NN classifier with leave-one-out cross-validation and performs forward selection and backward elimination on a user-specified synthetic dataset.

---

#### Task 2: Feature Selection on the Real Dataset

```bash
python preprocessing.py
```

Downloads and preprocesses the Wisconsin Diagnostic Breast Cancer dataset, including label encoding and z-score normalization.

```bash
python forward_backward_search.py
```

Runs forward selection and backward elimination on the preprocessed breast cancer dataset and reports the best feature subsets and accuracies.

```bash
python forward_ablation.py
```

Performs an ablation study by rerunning forward selection after removing the top three features from the original search.

---

#### Task 3: Dendrogram Clustering

```bash
python dendrogram.py
```

Generates a hierarchical clustering dendrogram for the Zoo dataset using binary feature representations and Jaccard distance.