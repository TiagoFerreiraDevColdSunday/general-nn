import keras

from keras import layers

# Building a long short term memory model
# NN to decide wether to keep, update or forget information
# A Recurrent NN, has memory cells and gates.
# Gates: Responsible to decide the flow, if they keep the memory / update ir or just forget it
# There are 3 gates:
# Forget Gate: Final value is the range of [0,1] 0 being completly forget, or 1 keep it fully
# Input Gate: How much information should be stored
# Output Gate: How much information should be the ouput as a hidden layer

# Candidate Cell (proposal of new information):
# Candidate =tanh(Wc⋅[ht−1​,xt​] + bc​)
#
# ht−1: Previous hidden layer
# bc: bias
# Wc: weight
# xt: current input
#
# How is the cell updated:
# Ct = ft ⊙ Ct-1 + it ⊙ Candidate
# 
# ft ⊙ Ct-1 -> how much old memory we keep
# it ⊙ Candidate -> how much new memory we keep
#
# ft -> forget gate
# it -> input gate

def build_lstm(seq_len):
    # Keras is a hight-level API to build NN
    # batch_size = seq_len (the number of samples)
    # 1 feature in this case our validation if it is malicious or not cancer
    inp = keras.Input(shape=(seq_len, 1))

    # First layer: LSTM
    # 64 is the number of neurons
    x = layers.LSTM(64)(inp)

    # Second Layer: Dropout (30% of neurons will go 0), prevents overfitting
    # Overfitting is when a Model knows the training data too well that ends up memorizing instead of learning
    # Results on Validation/Test data stagnate or decrease
    # If A and B have high Activation (For sigmoid have values close to 1, for RELU huge positive values), and we dont use
    # dropout the NN will memorize that by using A and B it's always results in class X, by using dropout A and B may get
    # deactivated during training and NN must find other ways to answer the the traning sample.¢
    x = layers.Dropout(0.3)(x)

    # Dense does a linear formula (x = Wx + b)
    # Then a relu which is a max(0,x)
    # 32 is the number of neurons
    x = layers.Dense(32, activation="relu")(x)
    
    x = layers.Dropout(0.3)(x)


    # For the final layer we want just 1 neurons to predict the final value since we're just want to predict a true/false value.
    out = layers.Dense(1, activation="sigmoid")(x)

    # Initialize the model
    model = keras.Model(inp, out)

    # Adam is the Algorithm to update the gradients. (See README.md)
    # 1e-3 is the learning rate, the learning rate shows the progress of the LLM's learning
    # If it's too small it's basically learning nothing, if it's too big it's unstable.
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )

    # If the LLM predicted correctly, the loss is small and the adam barely gets updates

    return model