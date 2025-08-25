# Basic terms

Samples = nº rows in a dataset
Features = nº of collumns

Statified Split important topics:

- Training data: the data that will be used to train your model

- Validation data: a "mock test" to check wether your data is ready to be tested

- Test data: the model will apply its knowledge on this data


Memorizing: The model knows how to produce the same values as the test data.

Learning: The model indetifies patterns on the training data and predicts not-seen data.

How does Validation helps avoiding Memorization? Each time a training is performed validation is used to check its accuracy, if the validation accuracy is low it means the model is still not ready to be tried on test samples. Each time a training is performed, the validation is used to adjust the weights for better predictions..

If we didn't have validation test data the model would just memorize the training data, being unable to predict data outside of the training.

- Weight: assist less common classes to be relevant during training that why their values are higher than common classes.

- Gradients purpose is to adjust Weights, so we an uncommon class appears the gradients push the weights to increase them.
# Long Short Term memory

- Gates
- Cells (what kind of cells exists in LSTM ?)

A cadidate cells is the purposed new information that can be updated.

# Gradients

- The gradient is a indicator that tells the LLM what will happen if you change the weight
    - If the gradient is positive and large, increase the weight will cause in a big progress loss.
    - While if the gradient is negative, increasing it will help with the progress. 
    - If near 0 it means the weight is in a good state.

    Usually the update rule of a weight goes by:
        Wnew​=Wold​−η⋅gradient

        η= learning rate 

# Optimizers
Optimizers use gradients, learning rates and can also apply other algorithms to update Weights.

## Adam
- Momentum 1: Mean of the previous gradients plus the current one.

- Momentum 2: Controls how much a gradient should increase or shrink.


## Batch Size
- How many samples you will send the ML model in one go.

## Loss
Loss is a term for how wrong was the model compared to the srource of truth.


## Dense function
y=f(Wx+b)

x -> input vector, for our LSTM case is the 64 dimensions vector.
b -> bias
W -> the weight of the layer
f -> activation function (which we are using Relu so max(0,x))