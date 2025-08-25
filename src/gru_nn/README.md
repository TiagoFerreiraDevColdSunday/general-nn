Applying a GRU NN into an EuroMillions scneario

# Multi-hot encoding

- Instead of just giving the number [0,50] and [0,12], you present it as a binary vector:

[0,1]^62

- If 1 hit, if 0 didn't hit.

Example: [0,0,1,1,0,1,1]

T is the nº of vector we will be sending to the model.

The vector that we send to the model is as it follows:

[0,0,0..................1,0,0,1] (62 elements)




