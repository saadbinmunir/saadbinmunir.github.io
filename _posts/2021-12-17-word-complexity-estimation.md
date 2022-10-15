---
layout: default
title:  Word Complexity Estimation
tags: "Machine Learning"
author: "Dat Duong"
---

## Introduction
Predicting word complexity is a natural language processing task that given an input string and a specified substring, the model should make a prediction for the complexity of that substring. The data set were annotated by having human labellers grading the level of complexity fo the word phrase. The complexity of the a phrase is defined as the number of annotators who mark the phrase as complex over total number of annotators

## Load required packages
```python
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
```

## Read the training data
Let's import the data and take a quick look at them

```python
df = pd.read_csv("train_full.txt", header=None, delimiter = "\t")
df.columns = [
    "id", "sentence", "start_index", "end_index", "word", "num_native", "num_non_native",
    "num_native_yes", "num_non_native_yes", "complexity"
]

df.head()
```

## Preprocessing
* changes the letter to upper case, only keep ASCII characters. Replace all non-ASCII characters with "?
* create the new columns with the masked tokens is replaced by `<MASK>`

```python
df.sentence = df.sentence.str.upper().apply(lambda x: x.encode("ascii", errors='replace').decode())
df["masked_sentence"] = df.apply(
    lambda row: (
        row.sentence[:row.start_index] + 
        " ".join(["<MASK>"]*len(row.sentence[row.start_index: row.end_index].split(" "))) + 
        row.sentence[row.end_index:]
    ),
    axis=1
)

```
* Get all the tokens and build the tokenizer

```python
# tokens from both df["sentence"] and df["masked_sentence"]
tokens = set([])
for sentence in df.sentence.to_list():
    tokens.update(set(sentence.split(" ")))

for sentence in df.masked_sentence.to_list():
    tokens.update(set(sentence.split(" ")))


# for padding use '<PAD>' tokens
tokens.add("<PAD>")

# create a map from token to a unique number, which is the index of that tokens in the sorted `tokens` list
tokens = sorted(tokens)
token2idx = {token: i for i, token in enumerate(tokens)}
print("total number of tokens: ", len(tokens)

```

Tokenize all the input sentence and masked_sentence, pad all the sentence to the same length using the id of the `<PAD>` token. This must be done to df["sentence"] and df["masked_sentence"]

In addition, we also need to add one more feature to the model, which is the ratio between the length of the masked texts over the length of the sentence (without padding). The reason behind this is that a longer masked phrase tends to be more complex compare to a shorter one.

```python
# tokenize sentences and masked_sentences
tokenized_sentences = [[token2idx[token] for token in sentence.split(" ")] for sentence in df.sentence.tolist()]
tokenized_masked_sentences = [[token2idx[token] for token in sentence.split(" ")] for sentence in df.masked_sentence.tolist()]

# pad them to the same length
max_length = max([len(_) for _ in tokenized_sentences])
pad_token = token2idx["<PAD>"]
tokenized_sentences = np.array([(sentence + [pad_token]*max_length)[:max_length] for sentence in tokenized_sentences])
tokenized_masked_sentences = np.array([(sentence + [pad_token]*max_length)[:max_length] for sentence in tokenized_masked_sentences])

# calculate the length of the target words relative to the total length of the sentence
target_length = np.array((df.end_index - df.start_index)/(df.end_index - df.start_index).max())

# getting the label
y_true = np.array(df.complexity.tolist()
```

## Split the data for K-fold cross validations

In this part, we will use K=5. Therefore, the training data will be split to 5 equal parts. At the training steps, we will hold out 1 to be used as validation and the rest will be used for training

```python
def get_k_folds(K=5):
    """
    This function returns a list of K (train_dict, val_dict) tuples.
    """
    n = len(y_true)
    result = []
    for k in range(K):
        # mask for the validation
        mask = np.array([True if k/K < i/n < (k+1)/K else False for i in range(n)])

        # extract the validation set
        val_dict = {
            'tokenized_sentences': tokenized_sentences[mask],
            'tokenized_masked_sentences': tokenized_masked_sentences[mask],
            'target_length': target_length[mask],
            'y_true': y_true[mask]
        }

        # extract the training set
        train_dict = {
            'tokenized_sentences': tokenized_sentences[~mask],
            'tokenized_masked_sentences': tokenized_masked_sentences[~mask],
            'target_length': target_length[~mask],
            'y_true': y_true[~mask]
        }

        # append to the list of folds
        result.append((train_dict, val_dict))

    return result
```

## Build the data generator
```python

def data_generator(data_dict, batch_size):
    """
    This function is a generator for the data. On each batch, we should yield 2 inputs and 1 outputs
    This satisfies the requirements that the inputs must include 2 types of features. Here, the first
    feature is the texts themseves, the second the mentioned ratio
    """

    data_size = len(data_dict["tokenized_sentences"])

    i = 1
    while True:
        if i + batch_size > data_size:
            # shuffle the data after use up all the samples
            mask = np.arange(data_size)
            np.random.shuffle(mask)
            for key in data_dict:
                data_dict[key] = data_dict[key][mask]
            # reset the index
            i = 0

        # input_1 is the tokenized_setence
        input_1 = data_dict["tokenized_sentences"][i: i + batch_size]

        # input_2 is the tokenized_masked_sentence
        input_2 = data_dict["tokenized_masked_sentences"][i: i + batch_size]

        # input_3 is the ratio of the length of the target word relative to the length of the sentence
        input_3 = np.expand_dims(data_dict["target_length"][i: i + batch_size], axis=-1)

        # extracting the output
        output = data_dict["y_true"][i: i + batch_size]

        # increase the i index and yield the new batch of data
        i += batch_size

        # return a tuple of (X, y) where X is the list of the 3 inputs
        yield [input_1, input_2, input_3], output
```
## Build the model

* In this model, we will use an embedding layer to map each token to an features array of size 128. We will pad (or strip) the sentence to a predefined length of 50 tokens. The result after embedding lookup is an array of size (50, 128) for each sentence, (N, 50, 128) for each mini batch.
* We then send this through an LSTM layer and get only the final output. If the LSTM layer has the size of 512, the output will have the shape of (N, 512). Both sentence and masked sentence will need to be sent tthrough 1 single LSTM layer. 
* Next, we concatenate the LSTM outputs of both sentence and masked sentence. The output should be then (N, 1024)

* Last, we send this through a couple of Fully-Connected layers of sizes 512, 128, 1. Each layer should have 'relu' activation, except for the the last layer that has linear activation. 

```python
def build_model(max_length):

    # we must reset the keras session to remove old memory
    tf.keras.backend.clear_session()

    # embeddings to reduce the size of the input tokens size from ~6K to 128
    embeddings = tf.keras.layers.Embedding(
        input_dim=len(token2idx),
        output_dim=128,
    )

    input_1 = tf.keras.layers.Input(shape=(max_length,), dtype=tf.float32)
    input_2 = tf.keras.layers.Input(shape=(max_length, ), dtype=tf.float32)
    input_3 = tf.keras.layers.Input(shape=(1, ), dtype=tf.float32)

    # send both input_1 and _2 through the embedding layer
    x_1 = embeddings(input_1)
    x_2 = embeddings(input_2)


    # At this steps, x1 and x2 should already have the shape of (batch_size, max_length, 128)
    # define a bidirectional LSTM then send the two string through
    # the output shape of x_1, x_2 should be now (batch, 512)
    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256)
    )
    x_1 = lstm(x_1)
    x_2 = lstm(x_2)

    # send the input_3, through a Dense 32 layer
    x_3 = tf.keras.layers.Dense(32, activation='relu')(input_3)

    # concatenate x1, x2, x3 into one array. The output should be now [batch, 512 + 512 + 32]
    x = tf.keras.layers.Concatenate(axis=-1)([x_1, x_2, x_3])

    # Send through a Dense layer with "relu" activation, followed by a Dropout
    # layer to reduce overfitting. The output should be [batch_size, 256]
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Send through an another Dense layer with "relu" activation, followed by a Dropout
    # layer to reduce overfitting. The output should be [batch_size, 256]
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Finally, send through 1 Dense layer with 1 single unit and linear activation for regression
    y = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.models.Model(inputs=[input_1, input_2, input_3], outputs=[y])

    return model

model = build_model(max_length=max_length)
model.summary()
```

This is the output

```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 103)]        0           []

 input_2 (InputLayer)           [(None, 103)]        0           []

 embedding (Embedding)          (None, 103, 128)     618752      ['input_1[0][0]',
                                                                  'input_2[0][0]']

 input_3 (InputLayer)           [(None, 1)]          0           []

 bidirectional (Bidirectional)  (None, 512)          788480      ['embedding[0][0]',
                                                                  'embedding[1][0]']

 dense (Dense)                  (None, 32)           64          ['input_3[0][0]']

 concatenate (Concatenate)      (None, 1056)         0           ['bidirectional[0][0]',
                                                                  'bidirectional[1][0]',
                                                                  'dense[0][0]']

 dense_1 (Dense)                (None, 256)          270592      ['concatenate[0][0]']

 dropout (Dropout)              (None, 256)          0           ['dense_1[0][0]']

 dense_2 (Dense)                (None, 64)           16448       ['dropout[0][0]']

 dropout_1 (Dropout)            (None, 64)           0           ['dense_2[0][0]']

 dense_3 (Dense)                (None, 1)            65          ['dropout_1[0][0]']

==================================================================================================
Total params: 1,694,401
Trainable params: 1,694,401
Non-trainable params: 0
```
## Train the model
To train the model, we will feed the `train_dict` and `val_dict` to the generator, which handles batch generation. We will start by using Adam optimizer with a small learning rate of 1E-4. Because this is a regression problem, we will use `mean_absolute_error` as the loss function. The Early Stopping patience is set to 3 and the best model will be saved to file so that we can compare many different models using K-fold cross validation.

```python
def train_model(max_length, train_dict, val_dict, batch_size=32, model_name=None):
    """
    train the model and save the model with the given name
    """
    model = build_model(max_length=max_length)

    train_generator = data_generator(train_dict, batch_size=batch_size)
    val_generator = data_generator(val_dict, batch_size=batch_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mean_absolute_error'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        x=train_generator,
        epochs=50,
        validation_data=val_generator,
        steps_per_epoch=len(train_dict["tokenized_sentences"])//batch_size + 1,
        validation_steps=len(val_dict["tokenized_sentences"])//batch_size + 1,
        callbacks=[early_stopping],
        verbose=1
    )
    model.save(f"best_model_{model_name}.tf")
    return history
```
## K-fold Validation

In this session, we will do the k-fold cross valiation to select the best model
```python
history = []
for i, (train_dict, val_dict) in enumerate(get_k_folds(K=5)):
    current_history = train_model(max_length, train_dict, val_dict, batch_size=32, model_name=i)
    history.append(current_history)

```

The plot below shows the validation losses of different models trained on different set of train/validation splits

```python
best_val_loss = [np.min(history_.history['val_loss']) for history_ in history]
plt.title("K-fold Cross Validation")
plt.plot(np.arange(len(history)), best_val_loss, label="best validation loss")
plt.xticks(np.arange(len(history)), [f"model_{i}" for i in range(len(history))])
plt.ylabel("Validation Loss")
plt.legend()

```

![validation loss](https://github.com/datduonguva/cuddly-octo-succotash/blob/gh-pages/_posts/aa01.png?raw=true)

