# AT&T/Olivetti Facial Recognition with Machine Learning and Deep Learning

Using the [AT&amp;T Database of Faces](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces) dataset ([ORL]() Database of Faces), this Python script uses machine learning algorithms (Random Forest and Logistic Regression) and a convolutional neural network (CNN) for facial recognition.  The models work to identify the subject in the image from a known database of 40 individuals each with 10 images.

---

### TO DO

**Data Model Implementation (25 points)**

* [X] A Python script initializes, trains, and evaluates a model (10 points)
* [X] The data is cleaned, normalized, and standardized prior to modeling (5 points)
* [ ] ~~The model utilizes data retrieved from SQL or Spark (5 points)~~ I don't think either of these would work for our dataset of images.
* [X] The model demonstrates meaningful predictive power at least 75% classification accuracy or 0.80 R-squared. (5 points)

**Data Model Optimization (25 points)**

* [ ] The model optimization and evaluation process showing iterative
  changes made to the model and the resulting changes in model
  performance is documented in either a CSV/Excel table or in the Python
  script itself (15 points)
* [X] Overall model performance is printed or displayed at the end of the script (10 points)

**GitHub Documentation (25 points)**

* [X] GitHub repository is free of unnecessary files and folders and has an appropriate .gitignore in use (10 points)
* [X] The README is customized as a polished presentation of the content of the project (15 points)

**Group Presentation (25 points)**

* [ ] All group members speak during the presentation. (5 points)
* [ ] Content, transitions, and conclusions flow smoothly within any time restrictions. (5 points)
* [ ] The content is relevant to the project. (10 points)
* [ ] The presentation maintains audience interest. (5 points)

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

Using the sklearn library, the data is split into training and test sets and given a random_state=1.

##### Random Forest Classifier

***`<png>` balanced accuracy score***

***`<png>` confusion matrix***

***`<png>` classification report***

***`<png>` RFmisclassified_df***

##### Logistic Regression Model

***`<png>` balanced accuracy score***

***`<png>` confusion matrix***

***`<png>` classification report***

***`<png>` LRmisclassified_df***

### CNN Model

A random_state=1 is used again for the sake of comparison. The images are reshaped and normalized, and the labels are one-hot encoded. The CNN model is then compiled using the keras library and fit to the training data. The model's accuracy is then evaluated on the test data.

#### CNN Model Optimization

***The model optimization and evaluation process showing iterative changes made to the model and the resulting changes in model performance is documented in either a CSV/Excel table or in the Python script itself.***

### Visualizations

`<png>` True/Predicted visualizations

`<png>` graphs of comparative performance

`<png>` CNN model accuracy

`<png>` CNN model loss

### Note:

A random seed `<random_state=1>` was selected across all models to ensure reproducibility and to provide more valid and consistent comparison of their performance. Stratification was used to maintain the same distribution of classes in both training and testing sets. And a testing size of 20% was applied across all models enabling fair evaluation of each model's performance.

### References

[Using Random Forests for Face Recognition](https://notebook.community/mbeyeler/opencv-machine-learning/notebooks/10.03-Using-Random-Forests-for-Face-Recognition)

The above link contains an excerpt from the book [Machine Learning for OpenCV](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-opencv) by Michael Beyeler. While the Random Forest Classifier model in the script referenced in this ReadMe does not utilize CV, using Random Forest for this solution was inspired by the link.  Also of note from this excerpt, another way to import the dataset is to use is noted below. However, you do lose some control over image resolution and color mode customization.

```
from sklearn.datasets import fetch_olivetti_faces
dataset = fetch_olivetti_faces()
```

[What is MNIST? And why is it important? by SelectStar at medium.com](https://selectstar-ai.medium.com/what-is-mnist-and-why-is-it-important-e9a269edbad5)

The CNN model used heavily relies upon the code and walkthrough from this article, though the initial accuracy before some tweaks was abyssmal (0.025% accuracy).
