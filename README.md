# BKU DISSERTATION: CONTENT-BASED IMAGE RETRIEVAL USING CNN & TRIPLET LOSS

## Directory structure

``` structure
├── README.md
├── data
│   ├── interim
│   ├── processed
│   └── raw
│
├── models
│
├── notebooks
│
├── reports
│   └── figures
│
└── src

```

### Explain

- `/data`: store datasets
  - `/raw` store raw data which has been download or collected, never override of edit raw data.
  - `/processed` store final data, which ready for modeling
  - `/interim` store intermediate data that has been tranformed but not the final

- `/models`: trained and serialized models, model predictions, or model summaries

- `/notebooks`: Jupyter notebooks that will be run on Google Colab, never edit it locally

- `/reports`: Generated analysis as HTML, PDF, LaTeX, etc.

  - `figures`: Generated graphics and figures to be used in reporting

- `/src`: Source code for use in this project.