//this is my first github upload
//this is the algo for this classification

### Aim:
The aim of this task is to implement a **Naive Bayes Classification** algorithm from scratch using the **Bayes Theorem**. Specifically, this implementation applies the Naive Bayes classifier to a weather dataset, where the goal is to predict a target variable (e.g., whether it will be a good day for outdoor activity) based on several features (e.g., weather conditions such as rain, temperature, etc.).

The task includes:
- **Preprocessing**: Splitting the data into features and target.
- **Training**: Calculating necessary probabilities for class prediction.
- **Prediction**: Applying Bayes Theorem to predict the target class for given feature sets.
- **Accuracy**: Evaluating the model's performance using accuracy on both training and test sets.
- **Queries**: Making predictions for new, unseen data points.

### Algorithm:

1. **Preprocessing**:
   - Load the dataset into a DataFrame.
   - Split the data into **features** (X) and **target** (y).

2. **Train-Test Split**:
   - Divide the data into **training** and **testing** datasets using the `train_test_split` function.

3. **Naive Bayes Model**:
   - **Class Prior Calculation**:
     - Calculate the prior probabilities of each class (i.e., P(c)) from the target variable.
   - **Likelihood Calculation**:
     - For each feature in the dataset, calculate the likelihood of observing a particular feature value given the class (i.e., P(x|c)).
   - **Predictor Prior Calculation**:
     - Calculate the prior probability of each feature value (i.e., P(x)) based on the entire dataset.
     
4. **Prediction**:
   - For each query (set of feature values), apply **Bayes Theorem** to calculate the posterior probability of each class, given the feature values.
   - Choose the class with the highest posterior probability as the predicted class.

5. **Accuracy Evaluation**:
   - Calculate the accuracy of the model using the formula:
     \[
     \text{accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} \times 100
     \]
   - Evaluate accuracy on both the training and testing datasets.

6. **Queries**:
   - Make predictions on new, unseen data (queries) to see how the model performs with unknown inputs.

### Detailed Steps of Naive Bayes Classifier:

1. **Class Prior (P(c))**:
   - Calculate the probability of each class occurring in the dataset.

2. **Likelihood (P(x|c))**:
   - For each feature, compute the likelihood of each feature value occurring for a given class. This is essentially the conditional probability of the feature value given the class.

3. **Predictor Prior (P(x))**:
   - For each feature, calculate the overall probability of a feature value occurring in the entire dataset.

4. **Posterior Calculation**:
   - Using Bayes Theorem, calculate the posterior probability of each class for the given input features.
     \[
     P(c|x) = \frac{P(x|c) \cdot P(c)}{P(x)}
     \]
   - Choose the class with the highest posterior probability.

5. **Accuracy Calculation**:
   - Calculate the accuracy of the classifier on both the training and testing data by comparing the predicted labels with the actual labels.

### Naive Bayes Formula Breakdown:
- **Class Prior (P(c))**: Probability of each class in the target variable.
- **Likelihood (P(x|c))**: Conditional probability of a feature value given a class.
- **Predictor Prior (P(x))**: Probability of a feature value occurring across the entire dataset.
- **Posterior (P(c|x))**: Final probability of each class given the features (using Bayes' Theorem).

