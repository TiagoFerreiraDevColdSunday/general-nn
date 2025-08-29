import tensorflow as tf
# Gumbel top-k
# Similar to the next token distribution

# Gumbels adds noise (some randomness) to the final probabilities, therefore eventhough we pick the highest softmax.
# It's not greedy due to this randomness.

EPS = 1.20
# eps -> numerical stability so we don't get 
# shape is the dimensions of the gumbell vector
def _sample_gumbel(shape):
    
    U = tf.random(shape, minval=0.0, maxval=1.0)

    # Explanation:
    #
    # log(U) becomes negative so we make every value of U positive with negation..
    # By having every value positive, if we make a histogram out of those value we will get the curve e^-x
    #
    # The last -ln turns into gumbel noise
    return -tf.math.log(-tf.math.log(U + EPS) + EPS)



def gumbel_top_k(probs, k):

    # log is used to transform these probs into logits ready to be used for Gumbel
    # ex: [0.1, 0.3, 0.6] -- (applying log) --> [-2.3, -1.2, -0.5]
    # The 1e-20 is to avoid 0.
    logits = tf.math.log(tf.cast(probs, tf.float32) + EPS)

    # shape -> gives the dimensions of logits
    g = _sample_gumbel(tf.shape(logits))


def sample_draw(main_prob_vec, star_prob_vec, n_main, n_stars):
    return