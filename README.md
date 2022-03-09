# DIA_noDIA
# There's no difference: Convolutional Neural Networks for transient detection without template subtraction

We present a Convolutional Neural Network (CNN) model for the separation of astrophysical transients from image artifacts, a task known as "real-bogus" classification, that does not rely completly on Difference Image Analysis (DIA) which is a computationally expensive process involving image matching on small spatial scales in large volumes of data. Because it does not train using the difference image.

We train to CNNs for the ``real-bogus'' classification:

1. Used as training data the template, search and difference imageas, called **DIA-based model**.
2. Used as training data the template and search images, called **noDIA model**.

By this study we intend to show that:

1. A ``real-bogus'' classifiers with high accuracies (97% and 92%) through the use of Convolutional Neural Networks, which avoid the process of feature extraction by using tha images as input data. The result of these classifiers coincide with the result found by other CNN models, like [Gieseke et al. (2017)](https://academic.oup.com/mnras/article/472/3/3101/4093080), [Cabrera-Vives et al. (2016)](https://ieeexplore.ieee.org/document/7727206) and [2017](https://iopscience.iop.org/article/10.3847/1538-4357/836/1/97).
2. As a proof of concept, reduce the cost of transient detection by showing that is possible to avoid the Difference Image Analysis (DIA).

## Data

We use data from the first season of the Dark Energy Survey (DES). This data was used to train the "real-bogus" classifier implemented by DES, [autoscan](https://portal.nersc.gov/project/dessn/autoscan/#) that is explain in [Goldstein et al., 2015](https://iopscience.iop.org/article/10.1088/0004-6256/150/3/82/pdf).

Some examples of the data:

![imagen](https://user-images.githubusercontent.com/51520204/157523530-64cde26a-2ca0-440d-9d28-7aa49851eb97.png)

The data have 50% of "bogus" type object, like the first two object in the image; and 50% of "real" type object, like the last one two.

## Scaling and Normalization

Because we are are using a type of model (CNNs) that was desgin originally to, for example, classify dogs and cats. We have to pay extra atention to the values used as input data to train the models.

* For the difference images we standarized to have a mean μ of 0 and a standard deviation σ of 1.
* For the template and search images there a lot of extreme values, outside the 3σ interval. The following image show the pixel value distribution (in this case brightness) for the 3σ interval.
![imagen](https://user-images.githubusercontent.com/51520204/157537248-b673ec1f-bef2-4a59-bedb-965a0bcbf46c.png)

* To preserve this information and also the resolution given for the values inside the 3σ interval. We map the interval μ±3σ to 0-1, then the extreme values are greater than 1 or less than 0. The following image show the same four examples above, but after mapping.
![imagen](https://user-images.githubusercontent.com/51520204/157537360-50877fb7-01db-47b1-bc62-9fca87c19d18.png)

## Train data sets

We horizontally stacked the images to mimic closly what we, as human scanners, do to classify the images as "real" or "bogus". On the left there are some examples for the **DIA-based** model and on the right for the **noDIA** model (only template and search).
![imagen](https://user-images.githubusercontent.com/51520204/157544112-3ae77414-c08e-4c1c-8097-117389e28b02.png)

## CNNs architecture

For the **DIA-based** and **noDIA** models we designed two similar architectures. In this way we enable a easier and direct comparision of the perfomance of the CNN model to classify "bougs" and "real" objects. The parameters of hyperparameters are not optimized for the noDIA case. We just wanted to compare the making decision process of the CNNs models by just removing the difference image.

* **DIA-based**
  ![imagen](https://user-images.githubusercontent.com/51520204/157537686-3efcfeb8-ffcc-4c0d-a968-a3e374208e2b.png)
  
* **noDIA**
  ![imagen](https://user-images.githubusercontent.com/51520204/157538213-ac8b8001-4fd3-4106-9ce5-0a59de1b8c3d.png)
  
## Results and metrics

We have two models with very high accuracies for the "real-bogus" classification process
* Confusion matrix for the **DIA-based** model for the 20,000 images using for testing. Showing **97%** accuracy.
  ![imagen](https://user-images.githubusercontent.com/51520204/157538803-85e8704c-a36a-45a6-a6a7-5b50c888efb7.png)
  
* Confusion matrix for the **noDIA** model for the 20,000 images using for testing. Showing **92%** accuracy.
  ![imagen](https://user-images.githubusercontent.com/51520204/157539241-d28d23bb-81e0-4f50-8404-67a8c87d4f2c.png)

* Loss and accuracy curves for the **DIA-based** model on the left and **noDIA** on the right.
![imagen](https://user-images.githubusercontent.com/51520204/157539719-c16a5aa7-b36a-460e-b9a1-485d81e572c5.png)

## Exciting part!!!

We have two very good models that classify with high accuracy the "real-bogus" data that we used. We wanted to interpret those results via the feature importance analysis and provide some intuition about the differences between the **DIA-based** and **noDIA** models. We decided to explore the **Saliency maps** of the two models.

### What are Saliency maps?

Saliency maps quantify the importance of each pixel of an image in input to a CNN in the training process. 

They provide some level of interpretability through a process akin to feature importance analysis by enabling an assessment of which are the pixels the model relies on the most for the final classification.

If we are classifying dogs and cats, we would expect to have a lot of important pixels in the faces, eyes, body, ears, etc of the dogs and cats.

#### Expectations
* For the **DIA-based** case our expectation was to found a lot of important pixel in the difference image.
* For the **noDIA** case the expectation was less clear, beyond our experience (intuition) as human scanner in the "real-bogus" classification.

## Saliency maps for DIA-based model
![imagen](https://user-images.githubusercontent.com/51520204/157550542-f74d6b2e-7c2c-4aee-9d78-feb9ae5a495a.png)

## Saliency maps for the noDIA model
![imagen](https://user-images.githubusercontent.com/51520204/157550635-a3b32e07-d273-4c1e-895f-f54870b0af81.png)

## Quantitative exploration of the Saliency maps

We design a simple metric for the quantitative analysis. This one consists in the sum of all the saliency pixel in each of the segments of the input image (i.e., sum of only the pixel in the difference image),  normalized by the sum of all the pixel in the three segments.
If for example, for the final classification, the model relies only on the difference image, this segment would have a score of 1 and the template and the search a score of 0. If for the final classification, the model uses information of the three images for the final classification, each segment would have a score of ~0.333.

$I_{diff} = \frac{\sum{p_d}s_{p_d}}{\sum_{p}s_d}$

### DIA-based results

Confusion matrix reporting the proportion of transients for which the highest concentration of important pixels is found in the difference, search, or template
portion of the input image for the **DIA-based** model.

![imagen](https://user-images.githubusercontent.com/51520204/157552979-8f5ccf79-3f20-400b-bf95-bfd861fcea57.png)

In general the **DIA-based** model relies more on the information in the difference (D) image. For the "real" object correctly classify as "real" (the dark blue square), for 90% of them the model used more the information in the difference image, 9% more the information in the search, and 1% in the template.


### noDIA results

Confusion matrix reporting the proportion of transients for which the highest concentration of important pixels is found in the difference, search, or template
portion of the input image for the **noDIA** model.

![imagen](https://user-images.githubusercontent.com/51520204/157552822-abc1667a-8901-4342-97fd-4413d0064f68.png)


In general the **noDIA** model relies more on the information in the template (T) image. For the objects classify as "real", correct or incorrect (dark blue and light orange squares), the ratio between the template and the search was less than for the objects classify as "bogus".


# Packages requires

Pandas, Numpy, Keras, TensorFlow, Matplotlib, Seaborn.

All the code that supports the analysis presented here is available on a dedicated GitHub repository







