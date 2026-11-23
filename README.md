# PRML Lab 2: Correlation-Based Feature Reduction

A Google Colab notebook that writes the Pearson correlation coefficient from scratch, builds a correlation matrix, and uses it to find and drop redundant features. The same method is applied to three datasets: an indoor-positioning dataset, a housing price dataset and an advertising sales dataset.

<a href="https://colab.research.google.com/github/ayush1011-gsv/PRML2_Lab2/blob/main/PRML_LAB2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Contents

1. [Method](#method)
2. [Part 1: Indoor positioning data](#part-1-indoor-positioning-data)
3. [Part 2: Housing data](#part-2-housing-data)
4. [Part 3: Advertising data](#part-3-advertising-data)

## Datasets

| Part | File(s) | Target column(s) |
|------|---------|------------------|
| 1 | `trainingData.csv` and `validationData.csv` | `LONGITUDE`, `LATITUDE`, `FLOOR`, `BUILDINGID` |
| 2 | `Housing.csv` | `price` |
| 3 | `advertising.csv` | `Sales` |

Part 1 appears to be the UJIIndoorLoc Wi-Fi indoor localization data, where each row is a set of Wi-Fi signal strengths and the targets are the location of the measurement.

## Requirements

- Python 3.8 or later
- `numpy`, `pandas`, `seaborn` (and `matplotlib`, which seaborn uses)

Colab already includes all of them. To run locally:

```bash
pip install numpy pandas seaborn matplotlib jupyter
```

## How to run

1. Click the **Open in Colab** badge above.
2. Upload the CSV files from the table above to the Colab session so they are in `/content/`.
3. Choose **Runtime → Run all**.

---

## Method

All three parts follow the same steps.

**1. Pearson correlation from scratch.** For two columns `x` and `y` with `n` values:

```
r = cov(x, y) / (std(x) * std(y))
```

where the covariance and the standard deviations both divide by `n − 1`. The code uses these helper functions:

| Function | What it does |
|----------|--------------|
| `mean(values)` | Average of a list |
| `std_dev(values, avg)` | Sample standard deviation |
| `pearson_correlation(x, y)` | Correlation of two columns. It returns 0 if the lengths differ, if there is only one value, or if a column is constant (which would divide by zero) |
| `calculate_correlation_matrix(df)` | Correlation of every pair of columns, stored as a nested dictionary and then turned into a DataFrame |

**2. Absolute values.** The matrix is converted to absolute values so that strong negative correlations are treated the same as strong positive ones.

**3. Upper triangle.** Only the part of the matrix above the diagonal is kept, so each pair of columns is checked once and a column is never compared with itself.

**4. Dropping.** With a threshold of **0.85**, a column is marked for removal if its correlation with any earlier column is above the threshold. Of two highly correlated features, only the later one is dropped.

**5. Protecting the target.** Target columns are removed from the drop list, so the label is never discarded.

**6.** The marked columns are removed and the shapes of the original and reduced data are printed.

---

## Part 1: Indoor positioning data

- The two CSV files are loaded and stacked into one DataFrame with `pd.concat(..., ignore_index=True)`.
- The correlation filter is applied to every column. The targets `LONGITUDE`, `LATITUDE`, `FLOOR` and `BUILDINGID` are kept even if they would be dropped.
- The notebook prints the number of redundant features dropped and the shape of the data before and after.

This dataset has several hundred columns, and most of them are Wi-Fi signal columns, so the number of column pairs is very large. The from-scratch correlation function recomputes means and standard deviations for every pair, so it is slow here. `df.corr()` gives the same result much faster.

## Part 2: Housing data

**Preprocessing.** Text values are converted to numbers so that correlations can be computed:

| Original value | Number |
|----------------|--------|
| `yes` | 1 |
| `no` | 0 |
| `furnished` | 1 |
| `semi-furnished` | 0.5 |
| `unfurnished` | 0 |

**Result.** No pair of columns goes above the 0.85 threshold, so nothing is dropped. The strongest relationships in the printed matrix are:

| Column | Correlation with `price` |
|--------|--------------------------|
| `area` | 0.536 |
| `bathrooms` | 0.518 |
| `airconditioning` | 0.453 |
| `stories` | 0.421 |
| `parking` | 0.384 |
| `bedrooms` | 0.366 |
| `prefarea` | 0.330 |
| `mainroad` | 0.297 |
| `guestroom` | 0.256 |
| `basement` | 0.187 |
| `hotwaterheating` | 0.093 |

Among the features themselves, the highest correlation is only 0.409 (`bedrooms` and `stories`), so the features carry mostly separate information. A heatmap of the matrix is also drawn.

## Part 3: Advertising data

The data has 200 rows and four columns: `TV`, `Radio`, `Newspaper` and `Sales`.

| | TV | Radio | Newspaper | Sales |
|---|----|-------|-----------|-------|
| **TV** | 1.000 | 0.055 | 0.057 | 0.901 |
| **Radio** | 0.055 | 1.000 | 0.354 | 0.350 |
| **Newspaper** | 0.057 | 0.354 | 1.000 | 0.158 |
| **Sales** | 0.901 | 0.350 | 0.158 | 1.000 |

**Result.**
- `TV` and `Sales` are correlated at 0.901, which is above the threshold. This is the only pair above 0.85, and it involves the target, so `Sales` is protected and no feature is dropped (`Features to drop: []`).
- The final data keeps its shape of `(200, 4)`.
- The three advertising features are only weakly related to each other (the highest is `Radio` and `Newspaper` at 0.354), so none of them is redundant.
- `TV` is the strongest predictor of `Sales` (0.901), followed by `Radio` (0.350) and `Newspaper` (0.158).

---

## Notes

- Correlation only measures **linear** relationships, so a pair that is related in a non-linear way can pass the filter.
- The result depends on column order, because the later column of a correlated pair is the one that gets dropped.
- A feature that is strongly correlated with the **target** is useful, not redundant. That is why the target column is excluded from dropping.

## Project structure

```
.
├── README.md
├── PRML_LAB2.ipynb
├── trainingData.csv
├── validationData.csv
├── Housing.csv
└── advertising.csv
```
