---                                                                                                 
layout: default                                                                                     
title: Graph Attention Network Implementation with Tensorflow
tags: "Machine Learning"                                                                            
author: "Dat Duong"                                                                                 
---

# Graph Attention Network

Graph Neural Networks (GNNs) were introduced in Gori et al. (2005) and Scarselli et al. (2009) as a generalization of recursive neural networks that can directly deal with a more general class of graphs, e.g. cyclic, directed and undirected graphs. GNNs consist of an iterative process, which propagates the node states until equilibrium; followed by a neural network, which produces an output for each node based on it states.

There have been many approaches to GNNs: spectrial approaches that work with spectial representations of graphs by calculating the eigendecomposition of graph's Laplacian;  non-spectrial approaches that work directly on graph's by learning weight matrices for each convolutional layers.

In this short tutorial, we will take a look at the way that attention mechanisms can be used for node classification task for graph-structured data.


## Graph Attenion Layer

In this section, we will describe how a Graph Attention Layer work.

The input to the layer is a set of node features $$\textbf{h} = \{\vec{h_1}, \vec{h_2}, \vec{h_3}, ..., \vec{h_N} \}$$, $$\vec{h_i} \in \mathcal{R}^{F}$$ where $$N$$ is the number of nodes, $$F$$ is the size of node feature. The output of the Graph Attention Layer will be another set of node features $$\textbf{h'} = \{\vec{h'_1}, \vec{h'_2}, \vec{h'_3}, ..., \vec{h'_N} \}$$, $$\vec{h'_i} \in \mathcal{R}^{F'}$$  where $$F$$ and $$F'$$ can be different.

A learnable weight matrix $$\textbf{W} \in \mathcal{R}^{F'\times F}$$ is applied to every node followed by a self-attention. The attention weight representing the importance of node $$i$$ on node $$j$$:

$$ e_{ij} = a(\textbf{W}\vec{h_i}, \textbf{W}\vec{h_j}) $$

where $$a$$ is simply a fully connected layer with LeakyReLU activation whose input is the concatenation of $$\textbf{W}\vec{h_i}$$ and $$\textbf{W}\vec{h_i}$$ and whose output is a scalar. To make the coefficients comparable across different nodes on the network, we need to normalize these coefficients among a node's neighbors using softmax function:

$$ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

Once $$\alpha_{ij}$$ is obtained, the aggregated features $$\vec{h'}$$ can be calculated by:

$$ \vec{h_i}' = \sigma (\sum_{j \in \mathcal{N}_i}) \alpha_{ij} \textbf{W}\vec{h_j}$$.

The idea of multi-head attention can also be used here. By applying the attention mechansim above $$K$$ times and averaging the result, 

$$ \vec{h_i}' = \sigma(\frac{1}{K}\sum_{k=1}^K) \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \textbf{W}^k\vec{h_j})$$

where the $$k$$ index denotes the parameters for the $$k$$-th attention.


Having defined this attention layer, we can implement this using Tensorflow/keras:


## Tensorflow/Keras implementation

### Implement the GraphAttentionLayer
First, we will define a class that is a subclass of `tf.keras.layers.Layer` for `Tensorflow` to handle autograd. The main parameter to the constructor is the input_dimension $$F$$, output dimension $$F'$$, number of attention head $$K$$:

```python
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, input_dimension, output_dimension, n_heads, activation, adjacent_matrix):
        super(GraphAttentionLayer, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_heads = n_heads
        self.activation = activation

        # take into account the self attention
        self.adjacent_matrix = adjacent_matrix + np.eye(len(adjacent_matrix))

```

The `activation` and `adjacent_matrix` will be used to aggregate the features for each nodes based on their neighbors. It can also be seen that when we will also add an diagnonal elements to the adjacent matrix to represent the self-connection of a node.


To represent the kernel weights $$\textbf{W}$$, we will add the weights to layer:

```python
        self.kernel_weights = [
            self.add_weight(
                f"kernel_w_{i}",
                shape=[self.input_dimension, self.output_dimension],
                dtype=tf.float32,
                regularizer=tf.keras.regularizers.L2(l2=5e-2)
            )
            for i in range(self.n_heads)
        
```
where the weights must be defined for all $$n_heads$$ different attention heads. The kernel weights have shape of (`input_dimentions`, `output_dimension`).

Next, we need to add the weights of the attention layers, whose input is the concatenation of the two feature vectors of shape `output_dimension` and the output is a scalar indicating the importance of one feature vector to the other. We also need to define the attention weights for `n_heads` different heads.

```python
        self.attention_weights = [
            self.add_weight(
                f"attention_weight_{i}",
                shape=[2*self.output_dimension, 1],
                dtype=tf.float32,
                regularizer=tf.keras.regularizers.L2(l2=5e-2)
            )
            for i in range(self.n_heads)
```

At this point, we are ready to define the forward function for the graph attention layers:

```python
   def call(self, inputs):
        # define an array that will be used to store output from all attention head
        output_features = tf.zeros([tf.shape(inputs)[0], self.output_dimension], dtype=tf.float32)

        # loop through all heads
        for i in range(self.n_heads):

            # first, transform the input features to hidden features
            transformed_features = tf.matmul(inputs, self.kernel_weights[i])

            batch_size = tf.shape(inputs)[0]

            # We want to calculate the attention of each features to all other feature.
            e_i = tf.repeat(transformed_features, batch_size, 0)
            e_j = tf.tile(transformed_features, [batch_size, 1])

            # concatenate the 2 feature vectors into 1
            e_features = tf.concat([e_i, e_j], axis=1)

            # multiply with the attention weights to get a single scalar
            e_features = tf.matmul(e_features, self.attention_weights[i])
            e_features = tf.keras.layers.LeakyReLU(alpha=0.2)(e_features)
            e_features = tf.reshape(e_features, (batch_size, batch_size))
            
            # scale the attention by using softmax activate.
            # Note: here, the attention is only non-zero for nodes
            # that has connections with each other
            e_features = tf.exp(e_features)*self.adjacent_matrix
            attention_coefficients = e_features/tf.reduce_sum(e_features, axis=1, keepdims=True)
            transformed_features = tf.matmul(attention_coefficients, transformed_features)
            
            # accumulate the transformed_features to the output_features
            output_features += transformed_features
        
        # averaging from all heads
        output_features /= self.n_heads
        
        # activation
        output_features = tf.keras.layers.Activation(self.activation)(output_features)

        return output_features
```

A trick that I used here to calculate the attention between each feature vector to all other feature vectors without using 2 nested for-loops is by using `tf.tile` and `tf.repeat`. Let's assume that we have a list `a = [1, 2, 3, 4, 5]`. To create a combination of each element in `a` to all other elements in `a`, we can write:

```python
combinations = []
for i in range(len(a)):
    for j in range(len(a)):
        combinations.append([i, j])

for item in combinations:
    print(item)
```

This should give the output
```python
[1, 1]
[1, 2]
[1, 3]
[1, 4]
...
[5, 1]
[5, 2]
...
[5, 5]
```

To achieve the same goal using Tensorflow, we can do:
```python
a_i = tf.repeat(a, 5)         # a_j is now [1, 1, 1, 1, 1, 2, 2, 2, 2..., 5, 5, 5, 5, 5]j
a_j = tf.tile(a, [6])         # a_i is now [1, 2, 3, 4, 5, 1, 2, ..., 1, 2, 3, 4, 5]
combinations = tf.stack([a_i, a_j], axis=1)
print(combinations)
```
producing:
```python
array([[1, 1],
       [1, 2],
       [1, 3],
        ...
       [5, 3],
       [5, 4],
       [5, 5]],
```
### Build 2-layer Graph Attention Network

Having defined the GraphAttentionLayer, the next steps of building the full model is pretty straightforward. The structure of the model is as below:

```python
def build_model():

    # input layer receiving the node's features
    input_ = tf.keras.layers.Input(
        shape=X.shape[1],
        batch_size=X.shape[0],
        dtype=tf.float32
    )

    # send through the first Graph Attention layer where input dimension is the length
    # of node's feature vectors. Output dimension is subjected to experiments
    x = tf.keras.layers.Dropout(0.5)(input_)
    x = GraphAttentionLayer(
        input_dimension=X.shape[1],
        output_dimension=8,
        n_heads=8,
        activation="elu",
        adjacent_matrix=A
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # send through the second Graph Attention layer where the output dimension is the
    # number of classes we want to predict
    output_ = GraphAttentionLayer(
        input_dimension=8,
        output_dimension=len(set(labels)),
        n_heads=1,
        activation="softmax",
        adjacent_matrix=A
    )(x)

    model = tf.keras.models.Model(inputs=input_, outputs=output_)
    return model
```

### Custom loss function
For this semi-supervised task, only 3.6% of the nodes have labels that are used for training (indicated by a binary mask). The loss function, which is categorical-crossentropy) must then be modified to take that into account. In addition, the accuracy of the prediction must be done for nodes having labels used for training and testing must be calculated separately as followed:

```python                              
def custom_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, 6))
    result = -tf.matmul(
        tf.reshape(tf.reduce_sum(y_true*tf.math.log(y_pred), axis=1), (1, -1)),
        tf.cast(tf.reshape(labelled_mask, (-1, 1)), tf.float32)
    )
    return result

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


### Results

By applying the Graph Attention Network to the Citeseer dataset, I got around 62% accuracy on the test nodes, which is pretty far away from the claimed performance of  72%.

For the complete implementation, please follow the link [Google Colab](https://colab.research.google.com/drive/11pHkn4oo2ufPPbWv35vHDNDeJSOSWogc?usp=sharing)

