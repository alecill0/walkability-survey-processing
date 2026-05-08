# Walkability Survey Processing

Python pipeline for processing, scoring, and clustering walkability perception survey data.

## Features

- Survey preprocessing
- Preference ranking normalization
- Weighted scoring system
- Socio-demographic encoding
- Dimensionality reduction with UMAP
- Hierarchical clustering
- Cluster validation metrics
- Automated CSV output generation

## Technologies

- Python
- pandas
- numpy
- scikit-learn
- scipy
- umap-learn
- matplotlib

## Methodology

Survey responses are transformed into normalized weighted scores
based on both:
- variable selection
- preference ranking

The weighting system used in this project was calibrated iteratively
during the clustering stage of the research workflow.

Different weighting combinations were evaluated to identify
configurations that produced better cluster separation
and consistency.

Clustering performance was assessed using:
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

The final configuration (30% selection / 70% ranking)
was chosen based on the overall clustering performance
across these metrics.

## Clustering Workflow

The clustering pipeline includes:

1. Data standardization using `StandardScaler`
2. Dimensionality reduction using UMAP
3. Hierarchical clustering
4. Cluster quality evaluation using internal validation metrics

Hierarchical clustering was implemented using:
- average linkage
- Chebyshev distance metric

## Project Structure

```text
data/       -> input survey data
outputs/    -> processed datasets and clustering outputs
src/        -> processing and clustering scripts
```

## Scripts

### `survey_processing.py`

Processes raw survey responses and generates weighted
walkability-related variables.

### `clustering_analysis.py`

Performs:
- dimensionality reduction
- hierarchical clustering
- cluster validation
- result exportation

## Outputs

The project generates:
- processed survey datasets
- clustered datasets
- validation metrics
- clustering visualizations