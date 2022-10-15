---                                                                                                 
layout: default                                                                                        
title: Graph Convolutional Network with Tensorflow
tags: "Machine Learning"                                                                            
author: "Dat Duong"                                                                                 
--- 
# Implement the Graph Convolutional Network with Tensorflow

## Introduction

In this short tutorial, we will implement the Graph Convolutional Network (GCN) with Tensorflow to solve the task of node classification where only a small subset of nodes have label.

Consider a graph of $$N$$ nodes and $$V$$ edges. Each node can have its own features $$X$$ of dimension $$D$$ ($$ X \in R^D$$ ). Let $$A$$ be the adjacency matrix ($$A \in R^{N\times N}$$), the structure of the network can be represented by a function $$f(X, A)$$ that propagates information on graphs.


According to the paper, the form of the functiof $$f(X, A)$$ is represented by:

$$
Z = f(X, A) = \text{softmax}(\hat{A} \text{ ReLU}(\hat{A}XW^{(0)})W^{(1)})
$$

where:

* $$\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$$,
* $$\tilde{A} = A + I_N$$ where $$I_N$$ is the identity matrix,
* $$\tilde{D}_{ii} = \sum_j \tilde{A}_{jj}$$,
* $$W^{0} \in R^{D\times H}$$ is the input-to-hidden weight matrix, which can be represented by a fully connected layer without bias in Tensorflow,
* $$W^{1} \in R^{H\times F}$$ is the hidden-to-output weight matrix, which can also be represented by a fully connected layer.


For a seme-supervised learning multiclass classification task, only a subset of the dataset has labels. The loss function is defined as:

$$
\mathcal{L}  = -\sum_{l \in \mathcal{Y}_L}\sum_{f=1}^F Y_{lf}\ln Z_{lf}
$$

where

* $$\mathcal{Y}_L$$ is the set of nodes that have labels. 


## Data

In this tutorial, we will use the Citeseer dataset, which is a citation network. The dataset has 3,327 nodes, 4,732 edges. Each node has a numerical feature array of size 3,703 and is classified into 1 of 6 availble classes. For this semi-supervise learning task, we will set the label rate at 0.036. This means that we will only use $$3.6\%$$ of the nodes for training and validating. The rest of the nodes will be used for evaluation.

To prepare the training data, first, we need to read the content file. Each line has the following format:

* the first integer is the node index
* the second to second to last integers are feature of the node
* the last word is the class of the node


What we are doing here is to create a dictionary whose key is the node index and whose value is a list of 2 elements: the node features and the node label

```python
with open("citeseer/citeseer.content", "r") as f:
    contents = [line.strip().split() for line in f.readlines()]
    contents = [[row[0], list(map(int, row[1:-1])), row[-1]] for row in contents]
    classes = sorted(set([row[2] for row in contents]))
    class_to_index = {class_: i for i, class_ in enumerate(classes)}

    # content is a dict whose key is the node's name, and whose values is a list of [features, label]
    contents = {row[0]: [row[1], class_to_index[row[2]]]  for row in contents}
        
```

Next, we need to read the citation network in the `cite` file:
```python
with open("citeseer/citeseer.cites", "r") as f:
    graph_rows = [line.strip().split() for line in f.readlines()]

    graph_rows = [line for line in graph_rows if line[0] in contents and line[1] in contents]

# get the list of start nodes and end_nodes
start_nodes, end_nodes = list(zip(*graph_rows))

# get the list of all nodes, and sort them in increasing order ( for consistency)
nodes = sorted(set(start_nodes + end_nodes))
nodes_to_index = {node: i for i, node in enumerate(nodes)}
```
Now, we can create the feature array $$X$$, the label array $$labels$$ and the adjacent matrix $$A$$:
```python
# create the feature array and format it to a numpy array
X = [ contents[node][0] for node in nodes]
X = np.array(X)

# create label arrays
labels = np.array([contents[node][1] for node in nodes])


# create the adjacent matrix:
A = np.zeros((len(nodes), len(nodes)))
for row in graph_rows:
    A[nodes_to_index[row[0]], nodes_to_index[row[1]]] = 1
    A[nodes_to_index[row[1]], nodes_to_index[row[0]]] = 1
```

As mentioned earlier, in this project will will only use 3.6% of the availabel labels for training and validation, the rest will be used for testing. To do this, we will generate a boolean mask as followed:

```python
# mask 
labelled_mask = (np.random.rand(len(A)) < label_rate) + 0
return A, X, labels, labelled_mask
```

## Build the GCN model

As we already have the adjacent matrix $$A$$,, we can calculate other quantities like:
* $$,\tilde{A} = A +  I_N$$,
* $$,\tilde{D}_{ii} =\sum_{j}\tilde{A}_{ij}$$,
* $$,D^{-1/2} = (D^{-1})^{1/2} = np.sqrt(D^{-1})$$,
* $$,\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$$,

These are implemented in Numpy as:
```python
A_tildle = A + np.eye(len(A))
D_tildle = np.diag(np.sum(A_tildle, axis=1))

# DD is used as notation for D^(-1/2)
DD = np.sqrt(np.linalg.matrix_power(D_tildle, -1))
A_hat = DD.dot(A).dot(DD).astype(np.float32)
```

As explained in the introduction, we need to transform the input to the output using:


$$
Z = f(X, A) = \text{softmax}(\hat{A} \text{ ReLU}(\hat{A}XW^{(0)})W^{(1)})
$$


In keras, this is written as

```python
input_ = tf.keras.layers.Input(
    shape=X.shape[1],
    batch_size=X.shape[0],
    dtype=tf.float32
)

x = tf.matmul(A_hat, input_, name='layer_input')
x = tf.keras.layers.Dense(16, activation='relu', name='layer_1', use_bias=False)(x)

x = tf.matmul(A_hat, x)
y = tf.keras.layers.Dense(6, activation='softmax', name='layer_2', use_bias=False)(x)

model = tf.keras.models.Model(inputs=input_, outputs=y)
```
### Define the loss function
For this semi-supervised task, the loss is aggregated across our labelled data. The boolean mask below helps setting all the loss of hold-out data to 0:
```python
def custom_loss(y_true, y_pred):
    y_true = tf.one_hot(y_true, depth=6)
    y_true = tf.reshape(y_true, (-1, 6))
    result = -tf.matmul(
        tf.reshape(tf.reduce_sum(y_true*tf.math.log(y_pred), axis=1), (1, -1)),
        tf.cast(tf.reshape(labelled_mask, (-1, 1)), tf.float32)
    )
    return result
```

Similarly, we can use the boolean mask to segregate the training data and validation data in measuing the accuracy of the trained model:
```python
def labelled_acc(y_true, y_pred):
    y_pred = tf.cast(tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=1), dtype=tf.int32)

    y_pred = y_pred[labelled_mask > 0.5]
    y_true = tf.cast(y_true[labelled_mask > 0.5], dtype=tf.int32)
    acc = tf.math.reduce_mean(tf.cast(y_true == y_pred, tf.float32))
    return acc

def unlabelled_acc(y_true, y_pred):

    y_pred = tf.cast(tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=1), dtype=tf.int32)
    y_pred = y_pred[labelled_mask < 0.5]
    y_true = tf.cast(y_true[labelled_mask < 0.5], dtype=tf.int32)

    acc = tf.math.reduce_mean(tf.cast(y_true == y_pred, tf.float32))
    return acc
```


### Train the model
At this point, we are ready to train the model:
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=custom_loss,
    metrics=[labelled_acc, unlabelled_acc]
)

model.fit(
    x=X,
    y=labels,
    batch_size=len(X),
    epochs=100,
    shuffle=False,
)
```

I can achieve around 0.62 acc for test dataset, compared to 0.7 as claimed in the paper:

```
Epoch 298/300
1/1 [==============================] - 0s 59ms/step - loss: 7.7962 - labelled_acc: 1.0000 - unlabelled_acc: 0.6135
Epoch 299/300
1/1 [==============================] - 0s 42ms/step - loss: 8.6400 - labelled_acc: 1.0000 - unlabelled_acc: 0.6123
Epoch 300/300
1/1 [==============================] - 0s 43ms/step - loss: 7.4976 - labelled_acc: 1.0000 - unlabelled_acc: 0.6123
```

## Summary
In this short tutorial, I have shown how one can impliment the GCN in tensorflow. I hope someone will find this helpful.
