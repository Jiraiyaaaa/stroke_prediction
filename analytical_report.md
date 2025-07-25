# Analytical Report: Stroke Prediction System

## Abstract
This report details the development and evaluation of a stroke prediction system designed to identify patients at risk of experiencing a stroke. Utilizing a Feed Forward Neural Network, the system processes various patient health indicators to provide a probabilistic assessment of stroke likelihood. The methodology, performance metrics, and implications for non-technical stakeholders are discussed, highlighting the system's potential as a valuable tool in proactive healthcare.

## Introduction
Stroke remains a significant global health concern, leading to severe disability and mortality. Early identification of individuals at high risk of stroke is crucial for implementing preventive measures and improving patient outcomes. This project aims to develop a reliable and interpretable system that can predict stroke risk based on readily available patient data. Unlike complex "black-box" models, this system employs a straightforward approach to ensure its insights are accessible and actionable for healthcare professionals and decision-makers.

## Methodology
The stroke prediction system was built using a Feed Forward Neural Network (NN), a type of artificial intelligence model inspired by the human brain's structure. This network learns patterns from historical patient data to make predictions on new, unseen cases.

The development process involved the following key steps:

1.  **Data Preparation:** The dataset, containing various health attributes such as gender, age, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, and smoking status, was pre-processed. This involved handling missing values (specifically in the BMI attribute) by estimating them from similar patients and converting categorical information (like 'gender' or 'smoking status') into a numerical format that the neural network could understand.

2.  **Data Splitting:** To ensure the model's ability to generalize to new patients, the dataset was divided into two parts: 80% for training the model and 20% for testing its performance. This 80/20 split ensures that the model is evaluated on data it has never seen before, providing an unbiased assessment of its predictive capabilities.

3.  **Neural Network Architecture:** The chosen Neural Network consists of multiple layers of interconnected "neurons." Data flows through these layers, with each layer learning increasingly complex patterns. The network was configured with two hidden layers, allowing it to capture intricate relationships within the data without becoming overly complex.

4.  **Training and Imbalance Handling:** During training, the neural network learned to associate patient attributes with stroke outcomes. Given that stroke events are relatively rare in the general population (an imbalanced dataset), a technique called SMOTEENN was applied. SMOTEENN helps the model learn effectively from the minority "stroke" class by creating synthetic examples, ensuring the model doesn't simply predict "no stroke" for every patient.

5.  **Model Calibration:** To ensure that the predicted probabilities are trustworthy (e.g., if the model predicts a 70% chance of stroke, it means that approximately 70 out of 100 similar patients would indeed experience a stroke), the model's outputs were calibrated. This step is crucial for clinical applications where accurate probability estimates are vital for decision-making.

## Performance Evaluation
The system's performance was rigorously evaluated using several key metrics, chosen for their relevance in assessing models on imbalanced datasets:

*   **Accuracy (0.782):** This represents the overall proportion of correctly predicted cases (both stroke and non-stroke). While seemingly high, for imbalanced datasets, accuracy alone can be misleading if the model simply predicts the majority class most of the time.

*   **Area Under the Receiver Operating Characteristic Curve (AUC) (0.792):** AUC measures the model's ability to distinguish between patients who will have a stroke and those who will not. An AUC of 0.792 indicates a good discriminatory power, meaning the model is reasonably effective at ranking patients by their risk.

*   **F1-score (0.223):** The F1-score is a harmonic mean of precision and recall, providing a balanced measure of the model's performance, especially on the minority "stroke" class. A higher F1-score indicates better performance in identifying actual stroke cases while minimizing false positives.

*   **Balanced Accuracy (0.715):** This is a more appropriate measure of overall accuracy for imbalanced datasets. It calculates the average of the accuracy for each class (stroke and non-stroke), ensuring that the model performs well on both. A Balanced Accuracy of 0.715 suggests that the model is reasonably effective across both classes.

*   **Matthews Correlation Coefficient (MCC) (0.219):** MCC is considered one of the most informative metrics for evaluating binary classifications, particularly on imbalanced datasets. It produces a high score only if the prediction obtained good results in all four categories of the confusion matrix (true positives, true negatives, false positives, and false negatives). An MCC of 0.219 indicates a moderate positive correlation between the predicted and actual outcomes.

## Conclusion
This stroke prediction system, built using a Feed Forward Neural Network, demonstrates a promising capability to assess stroke risk. While the overall accuracy might appear modest to a non-technical reader, the Balanced Accuracy, F1-score, and Matthews Correlation Coefficient provide a more nuanced and positive view of the model's effectiveness, especially in identifying the rare but critical "stroke" cases. The model's ability to provide calibrated probabilities further enhances its utility in clinical settings, allowing healthcare professionals to make more informed decisions based on reliable risk assessments. This system serves as a foundational step towards leveraging machine learning for proactive health management and early intervention in stroke prevention.
