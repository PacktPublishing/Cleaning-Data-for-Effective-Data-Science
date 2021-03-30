# Cleaning Data for Effective Data Science

This is the code repository for [Cleaning Data for Effective Data
Science](https://www.packtpub.com/product/cleaning-data-for-effective-data-science/9781801071291?utm_source=github&utm_medium=repository&utm_campaign=9781801071291),
published by [Packt](https://www.packtpub.com/?utm_source=github). It contains
all the supporting project files necessary to work through the book from start
to finish.

* **Paperback**: 498 pages
* **ISBN-13**: 9781801071291
* **Date Of Publication**: 30 March 2021

[<img src="./.other/cover.png" width="248">](https://www.amazon.com/gp/product/B08Z8GRYFY/)

## Links

* [Amazon](https://www.amazon.com/gp/product/B08Z8GRYFY/)
* [Packt Publishing](https://www.packtpub.com/product/cleaning-data-for-effective-data-science/9781801071291)

## About the Book

It is something of a truism in data science, data analysis, or machine learning
that most of the effort needed to achieve your actual purpose lies in cleaning
your data. Written in David’s signature friendly and humorous style, this book
discusses in detail the essential steps performed in every production data
science or data analysis pipeline and prepares you for data visualization and
modeling results.

The book dives into the practical application of tools and techniques needed
for data ingestion, anomaly detection, value imputation, and feature
engineering. It also offers long-form exercises at the end of each chapter to
practice the skills acquired.

You will begin by looking at data ingestion of data formats such as JSON, CSV,
SQL RDBMSes, HDF5, NoSQL databases, files in image formats, and binary
serialized data structures. Further, the book provides numerous example data
sets and data files, which are available for download and independent
exploration.

Moving on from formats, you will impute missing values, detect unreliable data
and statistical anomalies, and generate synthetic features that are necessary
for successful data analysis and visualization goals.

By the end of this book, you will have acquired a firm understanding of the
data cleaning process necessary to perform real-world data science and machine
learning tasks.

## Instructions and Navigation

All of the code for each chapter is within Jupyter Notebooks.

## Table of Contents

0. [Preface](Introduction.ipynb)
   1. Doing the Other 80% of the Work
   1. Types of Grime
   1. Nomenclature
   1. Typography
   1. Taxonomy
   1. Included Code
   1. Running the Book
   1. Using this Book
   1. Data Hygiene
   1. Exercises

1. [Data Ingestion – Tabular Formats](Data_Ingestion-Tabular.ipynb)
   1. Tidying Up
   1. CSV
      * Sanity Checks
      * The Good, The Bad, and The Textual Data
   1. Spreadsheets Considered Harmful
   1. SQL RDBMS
      * Massaging Data Types
      * Repeating in R
      * Where SQL Goes Wrong (And How to Notice It)
   1. Other formats
      * HDF5 and NetCDF-4
      * SQLite
      * Apache Parquet
   1. Data Frames
      * Spark/Scala
      * Panda and Derived Wrappers
      * Vaex
      * Data Frames in R (Tidyverse)
      * Data Frames in R (data.table)
      * Bash for Fun
   1. Exercises
      * Tidy Data from Excel
      * Tidy Data from SQL
   1. Denouement

2. [Data Ingestion – Hierarchical Formats](Data_Ingestion-Hierarchical.ipynb)
   1. JSON
      * NaN Handling and Data Types
      * JSON Lines
      * GeoJSON
      * Tidy Geography
      * JSON Schema
   1. XML
      * User Records
      * Keyhole Markup Language
   1. Configuration Files
      * INI and Flat Custom Formats
      * TOML
      * Yet Another Markup Language
   1. NoSQL Databases
      * Document-Oriented Databases
      * Key/Value Stores
   1. Denouement

3. [Data Ingestion – Repurposing Data Sources](Data_Ingestion-Other.ipynb)
   1. Web Scraping
      * HTML Tables
      * Non-Tabular Data
      * Command-Line Scraping
   1. Portable Document Format
   1. Image Formats
      * Pixel Statistics
      * Channel Manipulation
      * Metadata
   1. Binary Serialized Data Structures
   1. Custom Text Formats
      * A Structured Log
      * Character Encodings
   1. Exercises
      * Enhancing the NPY Parser
      * Scaping Web Traffic
   1. Denouement

4. [Anomaly Detection](Anomaly_Detection.ipynb)
   1. Missing data
      * SQL
      * Hierarchical Formats
      * Sentinels
   1. Miscoded Data
   1. Fixed Bounds
   1. Outliers
      * Z-Score
      * Interquartile Range
   1. Multivariate Outliers
   1. Exercises
      * A Famous Experiment
      * Misspelled Words
   1. Denouement

5. [Data Quality](Data_Quality.ipynb)
   1. Missing Data
   1. Biasing Trends
      * Understanding Bias
      * Internally Detectable
      * Comparison to Baselines
   1. Benford's Law
   1. Class Imbalance
   1. Normalization and Scaling
      * Applying a Machine Learning Model
      * Scaling Techniques
      * Factor and Sample Weighting
   1. Cyclicity and Autocorrelation
      * Domain Knowledge Cycles
      * Discovered Cycles
   1. Bespoke Validation
      * Collation Validation
      * Transcription Validation
   1. Exercises
      * Data Characterization
      * Oversampled Polls
   1. Denouement

6. [Value Imputation](Value_Imputation.ipynb)
   1. Typical-Value Imputation
      * Typical Tabular Data
      * Locality Imputation
   1. Trend Imputation
      * Types of Trends
      * A Larger Coarse Time Series
      * Non-Temporal Trends
   1. Sampling
      * Undersampling
      * Oversampling
   1. Exercises
      * Alternate Trend Imputation
      * Balancing Multiple Features
   1. Denouement


7. [Feature Engineering](Feature_Engineering.ipynb)
   1. Date/time fields
      * Creating Datetimes
      * Imposing Regularity
      * Duplicated Timestamps
   1. String fields
      * Fuzzy Matching
      * Explicit Categories
   1. String Vectors
   1. Decompositions
      * Rotation and Whitening
      * Dimensionality Reduction
      * Visualization
   1. Quantization and Binarization
   1. One-Hot Encoding
   1. Polynomial Features
      * Generating Synthetic Features
      * Feature Selection
   1. Exercises
      * Intermittent Occurrences
      * Characterizing Levels
   1. Denouement

8. [Closure](Closure.ipynb)
   1. What You Know
   1. What You Don't Know (Yet)

9. [Glossary](Glossary.ipynb)


## Related Products

* [Clean Code in Python - Second Edition](https://www.packtpub.com/product/clean-code-in-python-second-edition/9781800560215)
* [Machine Learning Using TensorFlow Cookbook](https://www.packtpub.com/product/machine-learning-using-tensorflow-cookbook/9781800208865)
* [Pandas 1.x Cookbook - Second Edition](https://www.packtpub.com/product/pandas-1-x-cookbook-second-edition/9781839213106)
