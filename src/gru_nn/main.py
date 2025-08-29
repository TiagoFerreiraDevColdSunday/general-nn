import numpy as np

from models import build_recurrent_nn
from gumbel import sample_draw

D = 62

# Number of rows per sample
T = 100

N_MAINS = 50
N_STARTS = 12

def main():
    model = build_recurrent_nn()


    X_dummy = np.random.randint(0, 2, size=(8, T, D)).astype("float32")


    # verbose -> how much info the model prints to the console
    numbers_p, star_p = model.predict(X_dummy, verbose=0) # shapes: 8 rows of 50 collumns / 8 ros of 12 collumns (8,50) and (8,12)

    # numers_p would be a 8 batch of a probs vector, something like: [0.05, 0.01,.........(50x)]
    # star_p is the same but with 12 dimensions for each batch
    mains, stars = sample_draw(numbers_p[-1], star_p[-1]) # demo sample doesn't matter if we pick the last vector or not.
