# Prediction-using-Machine-Learning-Algorithms

This reposistory describes application of fundamental machine learning algorithms on different types of datasets. There are four main algorithms applied in this report, which are
  1. Logistic regression: Modeled with and without regularization map a linear interpretation of the features to a categorical representation
  2. KNN with PCA:  K-nearest-neighbors is a model which takes in a newly observed point 𝑥, then calculates it’s distance from the other points within the datset using some chosen distance metric, returns a set of the k closest points 𝐾 to 𝑥. From here, in the classification setting, 𝑥 is classified to be the majority class of 𝐾. To use KNN and avoid the curse of dimensionality [7], it is crucial to implement principal component analysis (PCA) to map the dataset to a corresponding lower dimension.
  3. SVM with kernels: Support vector machines (SVM) with and without kernels learn a separating hyperplane between two distinct classes through the use of a margin based algorithm.
  4. Neural networks: Neural networks used within this project fall into two classes, feed forward neural networks and convolutional neural networks. In general, neural networks are highly non-linear statistical models. The feedforward neural network takes in non-linear combinations of features at each ”layer” through the use of activation functions 

These ML models are utilized to predict accuracy on three types of datasets.

Dataset Description
1) UCI Adult Dataset: The Adult dataset [1] is meant for the classification of income exceeding $50k per year
based on census data. There are both categorical and continuous features and missing values in the 48,000 records.
2) Wisconsin Breast Cancer Dataset: The Wisconsin breast cancer dataset [2] is based on digitized images
of breast cancer masses. The features are continuous (other than ID), and the target variable is a categorical
classification of cancerous versus non-cancerous. There are no missing records out of the 569 records available.
3) Fashion Mnist: The Fashion Mnist dataset [3] is a collection of 60,000 training images as 10,000 testing
images. Each image is a 28x28 image, which has an associated class labelled 0-9. Each pixel is considered to be
a continuous feature, and the target variable is class which is categorical.
