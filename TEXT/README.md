# The data: GoEmotions 
GoEmotions is corpus of 58k carefully curated comments extracted from Reddit, and labeled for 27 emotion categories or Neutral. The dictionary label - emotion is the following:

| Label | Emotion        |
|--------|----------------|
| 1      | admiration     |
| 2      | amusement      |
| 3      | anger          |
| 4      | annoyance      |
| 5      | approval       |
| 6      | caring         |
| 7      | confusion      |
| 8      | curiosity      |
| 9      | desire         |
| 10     | disappointment |
| 11     | disapproval    |
| 12     | disgust        |
| 13     | embarrassment  |
| 14     | excitement     |
| 15     | fear           |
| 16     | gratitude      |
| 17     | grief          |
| 18     | joy            |
| 19     | love           |
| 20     | nervousness    |
| 21     | optimism       |
| 22     | pride          |
| 23     | realization    |
| 24     | relief         |
| 25     | remorse        |
| 26     | sadness        |
| 27     | surprise       |
| 28     | neutral        |


The specific dataset used in this project is available at [TensorFlow TFDS](https://www.tensorflow.org/datasets/catalog/goemotions), where it is already divided into 3 subsets of data, for training, testing and validation:

- Size of training dataset: 43,410.
- Size of test dataset: 5,427.
- Size of validation dataset: 5,426.

<br>

# Dataset Metadata
The following table is necessary for this dataset to be indexed by search engines such as [Google Dataset Search](https://datasetsearch.research.google.com/).

| Property   | Value                  |
|------------|------------------------|
| name       | GoEmotions |
| description|GoEmotions contains 58k carefully curated Reddit comments labeled for 27 emotion categories or Neutral. The emotion categories are _admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise_. |
| sameAs    | [https://github.com/google-research/google-research/tree/master/goemotions](https://github.com/google-research/google-research/tree/master/goemotions) |
| citation  | [https://identifiers.org/arxiv:2005.00547](https://identifiers.org/arxiv:2005.00547) |
| provider  | <table><tr><th>Property</th><th>Value</th></tr><tr><td>name</td><td>Google</td></tr><tr><td>sameAs</td><td>[https://en.wikipedia.org/wiki/Google](https://en.wikipedia.org/wiki/Google)</td></tr></table> |