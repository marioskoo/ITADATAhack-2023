# ITADATAhack-2023

## About the dataset

The training data is a table with 51,968 rows and 4 columns. Each row represents a single law, and for each law it shows the following attributes:

* **CELEX_ID:** The unique alphanumeric identifier of the law.
* **Text:** The text of the law in plain text format.
* **Directory code:** The whole number between 1 and 20 which represents the main chapter of the EU in which the law is inserted.
* **Citations:** The normative references present in the law, reported in turn in plain text format.

Test data is provided unlabelled in a table of 12,292 rows and 3 columns (the target variable "Directory code" is missing).

The Dataset has been rebalanced by oversampling the laws belonging to underrepresented chapters, which is why there are duplicate CELEX_IDs.

## Labels

The labels represent the main chapter in which each law is included. At this stage, they are integers between 1 and 20.

## Objective

The objective of this project is to implement a classifier which, after any appropriate preprocessing of the text of the law and/or the references in natural language and using Machine Learning methods, predicts the chapter to which each law belongs, i.e. its Directory code. Each law is associated with a single Directory code, corresponding to the first level of hierarchy in the EU classification.

## About the event

This hackathon is organised by Open Data Playground in collaboration with:

* **Laboratorio di Big Data del CINI:** Italian center of expertise for the development of knowledge and technologies in the fields of Big Data and Data Science.
* **Università degli Studi Napoli Parthenope:** State university founded in 1920, it was born and developed in the center of the city of Naples, starting from the sciences related to the sea and international trade.
* **Università degli Studi di Bari “Aldo Moro”:** One of the largest universities in Italy, the University of Bari "Aldo Moro" offers a wide range of integrated and complete training opportunities of excellence.
* **High-Performance Computing for Artificial Intelligence (HPC4AI):** Open-access laboratory on High-Performance Computing (HPC) for artificial intelligence (AI), created by the University of Turin.

## Dependencies

The following dependencies are needed to run this project:

* Python 3.8+
* NumPy
* Pandas
* Scikit-learn
* Keras
* NLTK
* Matplotlib
* Seaborn

```bash
pip install pandas scikit-learn keras imbalanced-learn nltk tensorflow matplotlib seaborn
```

## How to run the project

1. Clone the repository.
2. Install the dependencies.
3. Run the main.ipynb:

## Students

* Attilio Di Vicino
* Valerio
* Mariano Forte
