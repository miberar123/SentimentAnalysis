# The data: GoEmotions 
GoEmotions is corpus of 58k carefully curated comments extracted from Reddit, and labeled for 27 emotion categories or Neutral. The dictionary label - emotion is the following:

| Label | Emotion        |
|-------|----------------|
|   0   | anger          |
|   1   | contemp        |
|   2   | disgust        |
|   3   | fear           |
|   4   | happy          |
|   5   | neutral        |
|   6   | sad            |
|   7   | surprise       |



The specific dataset used in this project is available at [TensorFlow TFDS](https://www.kaggle.com/datasets/geolek/grayscale-face-images), where it is already divided into 3 subsets of data, for training, testing and validation:

- Size of training dataset: 120,345.
- Size of test dataset: 40,634.
- Size of validation dataset: 39,896.

<br>

# Models
The models we have used to classify our dataset are as folloing ones:

- **CNN**: We have trained a cnn using the pytorch library, where our accuracy has been 0.42 using the model that best qualified of all those we have tested, (these models are stored in the folder models/models_cnn). In spite of being a low accuracy, this corresponds to the probelam we are dealing with since the task of classifying by an image 8 emotions is a task that a human would find difficult to perform.

- **CNN from Scratch**: In the from scartch model we have configured the layers we want ourselves. After testing two configurations since one of them gave overtraining in the model with the second one an accuracy of 0.38 has been achieved.

- **Pre_trained model**: In the pre-training model we have decided to choose the ResNet152 model since it was the one that gave the best results when we tested different models training only the cnn. To this model we have added some final layers to try to improve it. After making several tests varying the parameters of epchs and bath_size the best accuray that has been obtained has been of 0,14 since this model having been pretrained with other data has not learned as much as the other models.

In this project we have also developed a code for data augmentation. This technique is useful when we don't have enough data to generate more data by rotating the images themselves.

