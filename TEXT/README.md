# The data: GoEmotions 
GoEmotions is corpus of 58k carefully curated comments extracted from Reddit, and labeled for 27 emotion categories or Neutral. The dictionary label - emotion is the following:

| Label | Emotion        |
|-------|----------------|
|   0   | admiration     |
|   1   | amusement      |
|   2   | anger          |
|   3   | annoyance      |
|   4   | approval       |
|   5   | caring         |
|   6   | confusion      |
|   7   | curiosity      |
|   8   | desire         |
|   9   | disappointment |
|  10   | disapproval    |
|  11   | disgust        |
|  12   | embarrassment  |
|  13   | excitement     |  
|  14   | fear           |
|  15   | gratitude      |
|  16   | grief          |
|  17   | joy            |
|  18   | love           |
|  19   | nervousness    |
|  20   | optimism       |
|  21   | pride          |
|  22   | realization    |
|  23   | relief         |
|  24   | remorse        |
|  25   | sadness        |
|  26   | surprise       |
|  27   | neutral        |


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