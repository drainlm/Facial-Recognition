
# AT&T/Olivetti Facial Recognition with Machine Learning and Deep Learning

Facial recognition can be useful in practically any industry for security, verification, or VIP treatment purposes. This project's primary objective is accurately identify the subject in an image from the [ATT Database of Faces](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) dataset ([ORL]() Database of Faces). The dataset consists of 40 individuals each with 10 images.

This Python script experiments with three different models (Random Forest, Logistic Regression, and CNN) to asses their suitability, performance, and potential for real-world application. While Random Forest and Logistic Regression employ traditional machine learning algorithms, CNN's deep learning architecture is specifically designed for image-related tasks. By comparing these three models, we hope to gain a better understanding of the capabilities of machine learning and deep learning in the field of facial recogntion.

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

The data is extracted from a zip file named Faces.zip, converted to grayscale, and stored in a list with corresponding labels. The images are also encoded into a vector representation for easier processing and analysis. The original format is saved off for a 2d array for the CNN model and a flattened 1d array is saved off for the machine learning models.

### Exploratory Data Analysis

The entire dataset is visualized. Then a selected individual's set can be defined and displayed (this was very useful while developing the code as a reference point).

![1686291199020](image/README/1686291199020.png)

### Machine Learning Models

Using the sklearn library, the data is split into training and test sets. A random seed `<random_state=1>` was selected across all three models to ensure reproducibility and to provide more valid and consistent comparison of their performance. Stratification was used to maintain the same distribution of classes in both training and testing sets. And a testing size of 20% was applied across all models enabling fair evaluation of each model's performance. For Logistic Regression, the data is first scaled using `<'MinMaxScaler'>`.

#### 
    Random Forest Classifier

The Random Forest model demonstrates high precision, recall, and F1-scores for most classes (40 individuals), indicating its ability to correctly classify images of different individuals. The weighted average values show that the model's performance is consistent across all classes, considering the support values (2 images for each individual). The overall accuracy of 95% indicates the model's strong classification capability.

![1686289755452](image/README/1686289755452.png)

![1686289740846](image/README/1686289740846.png)

![1686289799393](image/README/1686289799393.png)

#### 
    Logistic Regression Model

The Logistic Regression model has high accuracy, sensitivity, and specificity in correctly identifying individuals from their images, making it highly reliable for facial recognition tasks in this dataset.

![1686289836608](image/README/1686289836608.png)

![1686289823449](image/README/1686289823449.png)

### CNN Models

The images are reshaped and normalized, and the labels are one-hot encoded. The CNN model is then compiled using the keras library and fit to the training data. The model's accuracy is then evaluated on the test data. Through an optimization process, each model version is saved individually to show iterative changes made to the model and the performance.

![1686289879804](image/README/1686289879804.png)

![1686289909231](image/README/1686289909231.png)

#### CNN Model Version 4

CNN Version 4 model has three convolutional layers (extracting relevant features from the images), uses a 3x3 kernel and the ReLU activation function. The number of filters is set to 32 each of the three layers and the 'he_uniform' kernel initializer is used to initialize the weights. After each layer, a max poolying layer with a 2x2 pool size is applied, which helps reduce the spatial dimensions of the extracted features so the most important information can be focused on. The output of the last max pooling layer flattens into a 1d vector. These flattened features are fed into two dense layers. The first dense layer consists of 100 units with the ReLU activation function and 'he_uniform' kernel initializer. The second dense layer consists of 40 units with the softmax activation funcation to produce probabilities for each of the 40 individuals. The model is compiled using the Adam optimizer, categorical cross-entropy loss funcation, and accuracy as the evaluation metric. 

![1686289962937](image/README/1686289962937.png)

In terms of performance, the CNN Version 4 model demonstrates reasonable performance, but further investigation and fine-tuning would be needed to improve its accuracy and performance on specific classes with lower scores. One possible reason for the poor performance could be that the dataset is relatively small with only 400 images. CNN models One thing to note is that the dataset is relatively small, consistenting of only 400 images. This may make it difficult for the CNN to learn complex patterns or to generalize them effectively.

![1686290518875](image/README/1686290518875.png)

![1686290498019](image/README/1686290498019.png)

![1686290476186](image/README/1686290476186.png)

### Comparative Visualizations

![1686290589307](image/README/1686290589307.png)

![1686290633690](image/README/1686290633690.png)

### Conclusions

Logistic Regression performed the best....

### PySpark

Based on an early hypothesis that there could be a correlation between misclassification and pixel intensity, the average pixel intensity for each subject's set of 10 images was calculated. A correlation was not found. However, this information was loaded into PySpark (primarily to satisfy part of a grading rubric for this project) and it is possible that with some additional digging and loading of data some relevant information could be retrieved for further exploration. 

![1686290672734](image/README/1686290672734.png)![1686290844570](image/README/1686290844570.png)

### References

[Using Random Forests for Face Recognition](https://notebook.community/mbeyeler/opencv-machine-learning/notebooks/10.03-Using-Random-Forests-for-Face-Recognition)

The above link contains an excerpt from the book [Machine Learning for OpenCV](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-opencv) by Michael Beyeler. While the Random Forest Classifier model in the script referenced in this ReadMe does not utilize CV, using Random Forest for this solution was inspired by the link.

[What is MNIST? And why is it important? by SelectStar at medium.com](https://selectstar-ai.medium.com/what-is-mnist-and-why-is-it-important-e9a269edbad5)

The initial CNN model in this script pulls directly from the code and walkthrough in this article, though the initial accuracy was very low (8.75%) prior to optimization.
