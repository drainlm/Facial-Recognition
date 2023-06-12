# AT&T/Olivetti Facial Recognition with Machine Learning and Deep Learning

**Group 4: Lisa Drain, Shoaib Farooqui, Dominic Marin, and Francisco Latimer**

Facial recognition can be useful in practically any industry for security, verification, or VIP treatment purposes. This project's primary objective is accurately identify the subject in an image from the [ATT Database of Faces](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) dataset (also known as the ORL Database of Faces). The dataset consists of 40 subjects (individuals) each with 10 images.

This Python script experiments with three different models (Random Forest, Logistic Regression, and a Convolutional Neural Network) to assess their suitability, performance, and potential for real-world application. While Random Forest and Logistic Regression employ traditional machine learning algorithms, CNN's deep learning architecture is specifically designed for image-related tasks. By comparing these three models, we hope to gain a better understanding of the capabilities of machine learning and deep learning in the field of facial recognition.

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

The entire dataset is visualized. Then a selected individual's set can be defined and displayed (this was very useful while developing the code as a reference point for performance).

![1686291199020](image/README/1686291199020.png)

Note on a lacking diversity: Of the 40 people, 35 are white men, 4 white women, and there is 1 Black man. Machine learning models learn patterns from the data they are trained on, and if the dataset primarily consists of a specific demographic, the model may not perform well when encountering individuals from diverse backgrounds. It is crucial to address these issues to avoid misidentification and discrimination. Due to the scope and time constraints of this project, this is not something that could be addressed, but it would be an important discussion for further exploration of performance and the implications of the model on this limited dataset.

### Machine Learning Models

Using the sklearn library, the data is split into training and test sets. A random seed `<random_state=1>` was selected across all three models to ensure reproducibility and to provide more valid and consistent comparison of their performance. Stratification was also used across all three to maintain the same distribution of classes in both training and testing sets. And a testing size of 20% was applied across all models as well, enabling fair evaluation of each model's performance.

Note: A testing size of 25% was tested across all three models, which resulted in lower accuracy across all three models (Random Forest 93.33%, Logistic Regression: 97.50%, and CNN_V9: 92.50%), likely due to the more limited training data. The code for this can be found in the repo under `<25_Percent_Training.ipynb>`.

#### Random Forest Classifier Results

The Random Forest model demonstrates high precision, recall, and F1-scores for most classes (40 individuals), indicating its ability to correctly classify images of different individuals. The weighted average values show that the model's performance is consistent across all classes, considering the support values (2 images for each individual). The overall accuracy of 95% indicates the model's strong classification capability.

![1686365743379](image/README/1686365743379.png)

![1686365727887](image/README/1686365727887.png)

![1686365774831](image/README/1686365774831.png)

#### Logistic Regression Model Results

Specific to Logistic Regression, the data is first scaled using `<'MinMaxScaler'>` and uses a `<'max_iter>` size of 2500, which iterates through until convergence (where the algorithm has reached a stable solution or the optimal values for the model's parameters). The Logistic Regression model has high accuracy, sensitivity, and specificity in correctly identifying individuals from their images, making it highly reliable for facial recognition tasks in this dataset.

![1686289836608](image/README/1686289836608.png)

![1686289823449](image/README/1686289823449.png)

### CNN Models

For CNN, the images are reshaped and normalized, and the labels are one-hot encoded. The CNN model is then compiled using the keras library and fit to the training data. The model's accuracy is then evaluated on the test data. Through an optimization process, each model version is saved individually to show iterative changes made to the model and the performance.

![1686365855847](image/README/1686365855847.png)![1686368125904](image/README/1686368125904.png)

![1686365883481](image/README/1686365883481.png)

Note on versions: While the README showcases nine versions of the CNN model, it is important to note that additional iterations were explored to further refine the model's performance. These additional iterations involved changes such as different activation functions (e.g., Leaky ReLU), alternative weight initialization techniques (e.g., Lecun and Xavier/Glorot), the inclusion of batch normalization, and attempts at data augmentation. Due to the computational expense, these additional attempts were not fully explored or incorporated into the final implementation.

#### CNN Model Version 9

CNN Version 9 model has three convolutional layers (extracting relevant features from the images), uses a 3x3 kernel and the ReLU activation function. The number of filters is set to 32 each of the three layers and the 'he_uniform' kernel initializer is used to initialize the weights. After each layer, a max poolying layer with a 2x2 pool size is applied, which helps reduce the spatial dimensions of the extracted features so the most important information can be focused on. The output of the last max pooling layer flattens into a 1d vector. These flattened features are fed into two dense layers. The first dense layer consists of 256 neurons and the second dense layer consists of 128 neurons with the ReLU activation function and 'he_uniform' kernel initializer. The final dense layer consists of 40 neurons with the softmax activation funcation to produce probabilities for each of the 40 individuals. The model is compiled using the Adam optimizer, categorical cross-entropy loss funcation, and accuracy as the evaluation metric.

![1686365928251](image/README/1686365928251.png)

##### CNN Model Version 9 Results

In terms of performance, the CNN Version 9 model demonstrates a high performance with an accuracy of 98.75% (misclassifying 2 images) and a test loss of 0.0464

![1686368371235](image/README/1686368371235.png)

![1686368348395](image/README/1686368348395.png)

![1686366059216](image/README/1686366059216.png)

### Comparative Visualizations

![1686366090356](image/README/1686366090356.png)

![1686366107306](image/README/1686366107306.png)

### Conclusions

Our results revealed that Logistic Regression achieved the highest performance, with an impressive accuracy of 100%. This model demonstrated excellent precision, recall, and F1-scores, indicating its reliability in correctly classifying images across different individuals. Logistic Regression can be considered a strong choice for facial recognition tasks in this dataset.

Although the CNN model achieved an accuracy of 98.75%, architecture required multiple versions of exploration before achieving this level of performance. Further investigation and further fine-tuning would still be needed to improve its accuracy and performance to achieve 100% accuracy. It's also worth noting that while CNN models are specifically designed for image-related tasks, our dataset of 400 images may have limited the model's ability to learn complex patterns, leading to slightly lower accuracy.

While Random Forest model achieved a lower accuracy of 95.00% accuracy, it is worth noting that this was achieved with the greatest of ease and was cheap computationally compared to the two other models. It demonstrated high precision, recall, and F1-scores for most classes, validating its effectiveness in correctly identifying individuals from their images. Random Forest can still be considered a robust alternative for facial recognition, when dealing with complex feature interactions.

The findings of this project highlight the strengths of Random Forest, Logistic Regression, and CNN models for facial recognition tasks, showcasing their high accuracy and reliability. Moreover, the exploration of the CNN model provides valuable insights into the potential of deep learning approaches for image recognition.

### A Note on Pixel Intensity and PySpark

Based on an early hypothesis that there could be a correlation between misclassification and pixel intensity, the average pixel intensity for each subject's set of 10 images was calculated. A correlation was not found. However, this information was loaded into PySpark (primarily to satisfy part of a grading rubric for this project) and it is possible that with some additional digging and loading of data some relevant information could be retrieved for further exploration. The section should be in Exploratory Data Analysis, however it is placed here to avoid confusion since it wasn't pursued further.

![1686290672734](image/README/1686290672734.png)![1686290844570](image/README/1686290844570.png)

### Note: 

For the misclassification visualization, you'll see a True image and a Predicted image. While the True image depicts the testing image, the Predicted image is just the first picture in the indvidual subject's set of 10 pictures. 

### References

[Using Random Forests for Face Recognition](https://notebook.community/mbeyeler/opencv-machine-learning/notebooks/10.03-Using-Random-Forests-for-Face-Recognition)

The above link contains an excerpt from the book [Machine Learning for OpenCV](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-opencv) by Michael Beyeler. While the Random Forest Classifier model in the script referenced in this ReadMe does not utilize CV, using Random Forest for this solution was inspired by the link.

[What is MNIST? And why is it important? by SelectStar at medium.com](https://selectstar-ai.medium.com/what-is-mnist-and-why-is-it-important-e9a269edbad5)

The initial CNN model in this script pulls directly from the code and walkthrough in this article, though the initial accuracy was very low (2.5%) prior to optimization.
