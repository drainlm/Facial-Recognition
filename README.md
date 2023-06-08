# AT&T/Olivetti Facial Recognition with Machine Learning and Deep Learning

Facial recognition can be useful in practically any industry for security, verification, or VIP treatment purposes. This project's primary objective is accurately identify the subject in an image from the [ATT Database of Faces](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) dataset ([ORL]() Database of Faces). The dataset consists of 40 individuals each with 10 images. 

This Python script experiments with three different models (Random Forest, Logistic Regression, and CNN) to asses their suitability, performance, and potential for real-world application. While Random Forest and Logistic Regression employ traditional machine learning algorithms, CNN's deep learning architecture is specifically designed for image-related tasks. By comparing these three models, we hope to gain a better understanding of the capabilities of machine learning and deep learning in the field of facial recogntion.

---

### Dependencies

* os
* zipfile
* numpy
* matplotlib
* pandas
* PIL (Python Imaging Library)
* sklearn
* collections
* tensorflow
* keras

### Loading the Data

The data is extracted from a zip file named Faces.zip, converted to grayscale, and stored in a list with corresponding labels.

### Exploratory Data Analysis

Here we visualize the images for a specific subject (useful in determining who the predicted subject was when viewing True vs. Predicted visualization). We also calculate the average pixel intensity for each subject's image (this was based on a hypothesis that there could be a correlation between misclassification and pixel intensity).

### Machine Learning Models

Using the sklearn library, the data is split into training and test sets. A random seed `<random_state=1>` was selected across all three models to ensure reproducibility and to provide more valid and consistent comparison of their performance. Stratification was used to maintain the same distribution of classes in both training and testing sets. And a testing size of 20% was applied across all models enabling fair evaluation of each model's performance.

##### 
    Random Forest Classifier

The Random Forest model demonstrates high precision, recall, and F1-scores for most classes (40 individuals), indicating its ability to correctly classify images of different individuals. The weighted average values show that the model's performance is consistent across all classes, considering the support values (2 images for each individual). The overall accuracy of 95% indicates the model's strong classification capability.

![1686194064477](image/README/1686194064477.png)

![1686194125584](image/README/1686194125584.png)

![1686196938521](image/README/1686196938521.png)

##### 
    Logistic Regression Model

The Logistic Regression model has high accuracy, sensitivity, and specificity in correctly identifying individuals from their images, making it highly reliable for facial recognition tasks in this dataset.

![1686194162581](image/README/1686194162581.png)

![1686194195317](image/README/1686194195317.png)

![1686196953585](image/README/1686196953585.png)

### CNN Model

The images are reshaped and normalized, and the labels are one-hot encoded. The CNN model is then compiled using the keras library and fit to the training data. The model's accuracy is then evaluated on the test data. Through an optimization process, each model version is saved individually to show iterative changes made to the model and the performance. 

![1686196274964](image/README/1686196274964.png)

![1686196325094](image/README/1686196325094.png)

CNN Version 4 model has three convolutional layers (extracting relevant features from the images), uses a 3x3 kernel and the ReLU activation function. The number of filters is set to 32 for each layer and the 'he_uniform' kernel initializer is used to initialize the weights. After each layer, a max poolying layer with a 2x2 pool size is applied, which helps reduce the spatial dimensions of the extracted features so the most important information can be focused on. The output of the last max pooling layer flattens into a 1d vector. These flattened features are fed into two dense layers. The first dense layer consists of 100 units with the ReLU activation function and 'he_uniform' kernel initializer. The second dense layer consists of 40 units with the softmax activation funcation to produce probabilities for each of the 40 individuals. The model is compiled using the Adam optimizer, categorical cross-entropy loss funcation, and accuracy as the evaluation metric. 

In terms of performance, the CNN Version 4 model demonstrates reasonable performance, but further investigation and fine-tuning is needed to improve its accuracy and performance on specific classes with lower scores.

![1686196073960](image/README/1686196073960.png)

![1686196115094](image/README/1686196115094.png)

![1686197247363](image/README/1686197247363.png)

### Comparative Visualizations

![1686196455191](image/README/1686196455191.png)

![1686196404442](image/README/1686196404442.png)

### References

[Using Random Forests for Face Recognition](https://notebook.community/mbeyeler/opencv-machine-learning/notebooks/10.03-Using-Random-Forests-for-Face-Recognition)

The above link contains an excerpt from the book [Machine Learning for OpenCV](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-opencv) by Michael Beyeler. While the Random Forest Classifier model in the script referenced in this ReadMe does not utilize CV, using Random Forest for this solution was inspired by the link.  

[What is MNIST? And why is it important? by SelectStar at medium.com](https://selectstar-ai.medium.com/what-is-mnist-and-why-is-it-important-e9a269edbad5)

The CNN model in this script pulls directly from the code and walkthrough in this article, though the initial accuracy was abyssmal prior to tweaking it by adding layers and epochs.
