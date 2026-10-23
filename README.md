# Penguin Clustering & PCA

Exploratory data analysis, clustering (K-Means, Ward hierarchical, DBSCAN) and PCA on a penguins dataset, written as a midterm (UTS) project for a Machine Learning course.

Everything runs from one script, in order: read data, EDA, preprocessing, choose K, profile clusters, compare methods, PCA, and a printed summary. Only the configuration block at the top needs editing to reuse it on another dataset.

> Note: the script's printed output and plot titles are in Indonesian (the course language). Code identifiers and this README are in English.

## Key results

| Item | Result |
| --- | --- |
| Data | 344 rows, 8 columns; 4 numeric features used (bill length, bill depth, flipper length, body mass) |
| Missing values | 109 cells; 86 rows (25%) miss at least one measurement; filled with the column median, 2 empty rows dropped |
| Outliers / skewness | 0 IQR outliers; largest skewness 0.45 (body mass) |
| Strongest correlation | flipper length vs body mass, r = 0.87 |
| Chosen K | 2 (silhouette 0.494, Davies-Bouldin 0.798) |
| Clusters | 219 smaller penguins (Adelie + Chinstrap) and 123 larger penguins (all Gentoo) |
| Ward vs K-Means | identical grouping (ARI = 1.000) |
| PCA | PC1 = 64.8%, PC2 = 19.8%, together 84.6% of variance; 2 components kept |
| K-Means in PCA space | silhouette 0.590, same clusters as original space (ARI = 1.000) |
| Agreement with species | ARI 0.655 at K = 2, 0.787 at K = 3 |

Main takeaway: the data separates cleanly along a body-size axis (PC1), which isolates Gentoo. Adelie and Chinstrap overlap on size and differ mainly in bill length (PC2), so the clusters are size groups, not species.

A modelling choice worth knowing: `island` and `sex` are excluded. When they were one-hot encoded, K-Means picked K = 9 and split by island and sex instead of body measurements (ARI against species only 0.284). `year` is ignored because it is an observation date, not a physical trait.

## Repository contents

| Path | Description |
| --- | --- |
| `clustering_pca_analysis.py` | The full analysis script |
| `penguins_uts.csv` | Dataset (344 rows, includes missing values) |
| `hasil/` | Figures (`.png`) and tables (`.csv`) produced by the script |
| `run_output.txt` | Console output of one full run |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
python clustering_pca_analysis.py
```

Run it from the repository folder so the relative path to the CSV resolves. Figures and tables are saved to `hasil/` (created automatically). It works offline and uses a fixed `RANDOM_STATE = 42`, so results are reproducible.

## Using your own data

Edit section 0 at the top of `clustering_pca_analysis.py`:

```python
FILE_PATH = "penguins_uts.csv"   # your CSV/Excel file; None = built-in wine demo
SEP = None                       # None = auto-detect; or ";" "," "\t"
KOLOM_ABAIKAN = ["year"]         # identifier/time columns to ignore
KOLOM_LABEL = "species"          # true class column, used only for comparison
ENCODE_KATEGORIK = False         # True = one-hot encode text columns, False = drop them
K_MIN, K_MAX = 2, 10
K_PILIH = None                   # set a number to force K
N_KOMPONEN = None                # None = components needed for >= 80% variance
```

It handles several encodings, semicolon separators and comma decimals.

## Method summary

1. **EDA:** types, missing values, duplicates, IQR outliers, skewness, histograms, boxplots, correlation matrix.
2. **Preprocessing:** median imputation, z-score standardisation (needed because body mass in grams would otherwise dominate distances and variance).
3. **Clustering:** K-Means for K = 2 to 10 scored by inertia, silhouette and Davies-Bouldin; cluster profiles (raw and z-score); Ward hierarchical and DBSCAN as comparisons; cross-tab and ARI against the species label.
4. **PCA:** explained variance, scree plot, loadings, 2D projection coloured by cluster and by species, biplot, and K-Means re-run in PCA space.

## Limitations

- K-Means assumes roughly spherical clusters and is sensitive to scale and outliers.
- PCA only captures linear structure; component names ("body size", "bill shape") are interpretations of the loadings.
- Silhouette 0.494 means real but not sharply separated groups.
- Median imputation on 90 cells narrows variance and can shift clusters; rerunning on complete rows only would test this.
- Cluster stability across seeds or resamples was not tested.

## Data source

The data looks like a modified copy of the Palmer Penguins dataset (Adelie, Chinstrap and Gentoo penguins, Palmer Archipelago, Antarctica, 2007 to 2009) with missing values, supplied by the course instructor. Check the original dataset's license and citation before redistributing.

## Requirements

Python 3.10+ with numpy, pandas, matplotlib, scikit-learn and scipy (see `requirements.txt`).
