Applying a GRU NN into an EuroMillions scneario

# Multi-hot encoding

- Instead of just giving the number [0,50] and [0,12], you present it as a binary vector:

[0,1]^62

- If 1 hit, if 0 didn't hit.

Example: [0,0,1,1,0,1,1]

T is the nº of vector we will be sending to the model.

The vector that we send to the model is as it follows:

[0,0,0..................1,0,0,1] (62 elements)



# To understand Gumbel...

# Exponential Distribution

density(x, λ) = λe^(-λx)

x -> unit time
λ -> number of events per unit time

if λ = 1 and x = 1, we would get density = 1 (would be at it's highest)
if x = 2 and λ = 1 we would get density = 0.135
So the longer the events happen per time unit (x increases and λ doesn't increase at the same ratio as x), the density decreases.

Density is the curve that shows how the probability is spread -> investigate.