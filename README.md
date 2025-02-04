# Neural Network Challenge 1

This repository contains the solution to the AI Boot Camp challenge **"Student Loan Risk with Deep Learning."** The goal of this challenge is to build a neural network model that predicts whether a student loan recipient will repay their loan. Additionally, the project discusses how to build a recommendation system for student loans.

# Overview

In this challenge, the following steps were completed:

1. **Data Preparation:**

- Loaded the student loan dataset from a CSV file.
- Explored the data to understand the features and target variable.
- Created feature (X) and target (y) datasets.
- Split the data into training and testing sets.
- Scaled the feature data using scikit-learn's StandardScaler.

2. **Model Building, Compilation, and Training:**

- Built a deep neural network model using TensorFlow’s Keras with two hidden layers.
- Compiled the model using the adam optimizer and binary_crossentropy loss function.
- Trained the model on the training data for 50 epochs.

3. **Model Evaluation and Saving:**

- Evaluated the model using the test data to determine loss and accuracy.
- Saved the trained model to a file named student_loans.keras.

4. **Predictions and Reporting:**

- Reloaded the saved model and used it to make predictions on the test dataset.
- Converted the model’s probability outputs to binary predictions.
- Generated a classification report comparing the predictions to the actual outcomes.

5. Discussion:

- Provided answers to questions regarding the design of a recommendation system for student loans, including the data needed, filtering methods, and real-world challenges.

# Files

- student_loans_with_deep_learning.ipynb
The main Jupyter Notebook containing all the code for data preparation, model training, evaluation, and predictions, as well as discussion responses.
- student_loans.keras
The saved Keras model file produced after training.
- README.md
This file, containing an overview of the project.

# How to Run

## In Google Colab:

1. Open Google Colab.
2. Upload the student_loans_with_deep_learning.ipynb file.
3. Run each cell in sequence to execute the data preparation, model training, and evaluation.

## In Jupyter Notebook:

1. Ensure you have Python installed along with the required libraries (tensorflow, pandas, scikit-learn).
2. Open the notebook (student_loans_with_deep_learning.ipynb) in Jupyter Notebook.
3. Run all cells sequentially to reproduce the results.

## Requirements

- Python 3.x
- TensorFlow
- Pandas
- Scikit-learn

You can install the required packages using pip:

    ``pip install tensorflow pandas scikit-learn``

# Notes

- This project is part of an AI boot camp challenge. The goal was to demonstrate data preprocessing, model building with TensorFlow, evaluation of a neural network, and considerations for building recommendation systems in the context of student loans.
- The discussion section at the end of the notebook provides insights into data collection, filtering techniques, and real-world challenges for recommendation systems in this domain.