# Movie Review Sentiment Classification 

This project aims to classify movie reviews into two categories: **positive** and **negative**, using a **Bag of Words (BoW)** approach combined with **pre-trained GloVe embeddings**. A deep learning model (a neural network) is used for text classification. The model is built using the **Keras** library with a simple architecture involving an embedding layer, followed by fully connected layers.

---

## Features

- **Preprocessing**: The text data is preprocessed using tokenization and padding to prepare it for input to the neural network.
- **Embedding Layer**: Uses **GloVe embeddings** (100-dimensional word vectors) for word representations to leverage pre-trained knowledge.
- **Model Architecture**: The model is built with an embedding layer, followed by dense layers, and a sigmoid output for binary classification (positive or negative sentiment).
- **Evaluation**: The model is evaluated based on accuracy and loss, with a validation dataset split.

---

## Project Structure

- **Classifying_movie_reviews_BoW.ipynb**: The main Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
- **glove.6B.100d.txt**: Pre-trained GloVe embeddings (100d version) used in the project (you need to download it separately).
- **README.md**: This file providing documentation and instructions for the project.

---

## Prerequisites

Before running the project, ensure you have the following:

- Python 3.x
- TensorFlow/Keras
- NumPy
- GloVe embeddings (100d version)

You can install the required Python packages using `pip`:

```bash
pip install tensorflow numpy
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
```

## Dataset
The dataset used for this project (imdb) consists of movie reviews labeled as either positive or negative.
The data is preprocessed for tokenization and padding before being input into the model. 
This project assumes the dataset has already been split into training and test sets

to download datasest: 
```
from pathlib import Path
import os
DATA_PATH=Path('./dat/')
DATA_PATH.mkdir(exist_ok=True)
if not os.path.exists('./dat/aclImdb'):
    !curl -O http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    !tar -xf aclImdb_v1.tar.gz -C {DATA_PATH}
```

## Models 
- Bag Of Word Model
- Vector Bag Of Word Model

## Results 
- Bag Of Word Model using TFIDF feature its accuracy is equal 85 %
- Vector Bag of Word Model: 
  1. learn embeding  (accuarcy = 83 %)
  2. using pretrained model (GloVe) without training embedding and using GloVe embeding matrix as it is  (accuracy = 71%)
  3. using pretrained model (GloVe) and learning emmbeding matrix in my dataset ( accuracy = 82 %)


