## Custom Perceptron Model for Binary Classification

This repository contains a custom implementation of the Perceptron learning algorithm for binary classification. The code demonstrates the step-by-step process of training, iterative refinement, testing, and accuracy calculation using a simple perceptron model.

### Overview

The provided code showcases the complete workflow of a Perceptron model for binary classification without relying on external libraries like scikit-learn. The repository includes:

- **Data Loading and Handling:** Demonstrates how to load and handle data using the Pandas library. The dataset consists of performance statistics of FIFA 2021 players, including attributes like pace, shooting, passing, and more.

- **Feature Extraction and Normalization:** Extracts relevant features from the dataset and normalizes them to prepare for analysis.

- **Target Data and Dimensionality:** Explores the dimensionality of the feature data and retrieves the target values for analysis.

- **Model Training:** Implements the Perceptron learning algorithm for model training. The code adjusts weights iteratively to minimize prediction errors.

- **Iterative Model Refinement:** Refines the trained model iteratively using the concept of epochs. The model is updated based on prediction errors for each data point.

- **Model Testing and Accuracy Calculation:** Tests the trained model using a separate testing dataset and calculates its accuracy. The accuracy is calculated as the ratio of correct predictions to the total number of samples in the testing dataset.

This repository provides a clear and insightful walkthrough of building a custom Perceptron model from scratch, enabling a deeper understanding of the fundamental concepts behind binary classification.

For more information and code details, please refer to the respective sections of the provided Jupyter Notebook.