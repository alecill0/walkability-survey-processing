# Walkability Survey Processing

Python pipeline for processing and scoring walkability perception surveys.

## Features

- Survey preprocessing
- Preference ranking normalization
- Weighted scoring system
- Socio-demographic encoding
- Automated CSV output generation

## Technologies

- Python
- pandas
- numpy

## Methodology

Survey responses are transformed into normalized weighted scores
based on both:
- variable selection
- preference ranking

The weighting system used in this project was calibrated iteratively
during the clustering stage of the research workflow.

were evaluated to identify configurations that produced better cluster separation and consistency.

Clustering performance was assessed using:
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

The final configuration (30% selection / 70% ranking)
was chosen based on the overall clustering performance across these metrics.


## Project Structure

```text
data/       -> input survey data
outputs/    -> processed outputs
src/        -> processing scripts