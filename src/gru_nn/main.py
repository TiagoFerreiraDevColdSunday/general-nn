import numpy as np
import pandas as pd
import keras

from gumbel import sample_draw_probs
from draw_utils import encode_draw, draw_to_freq_targets, make_windows, year_mask
from src.models import build_recurrent_nn
from constants import MAIN_COLS, STAR_COLS, D, T, N_MAINS, N_STARS, VAL_YEARS, TEST_YEARS

# File Setup
CSV_PATH = "../data/EuroMillions_numbers.csv"
SEP = ";"
DATE_COL = "Date"



def main():

    df = pd.read_csv(CSV_PATH, sep=SEP, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL, ascending=True).reset_index(drop=True)


    # Ensure every value is int
    for c in MAIN_COLS + STAR_COLS:
        df[c] = df[c].astype(int)

    encoded = np.stack([encode_draw(r) for _, r in df.iterrows()], axis=0)

    print(f"Encoded format: \n{encoded}")

    targets_main = []
    targets_star = []

    for _, r in df.iterrows():
        # The respective percentages
        p_n, p_s = draw_to_freq_targets(r)
        targets_main.append(p_n), targets_star.append(p_s)

    # np.stack -> turn into an array of arrays in this case
    targets_main = np.stack(targets_main)
    targets_star = np.stack(targets_star)

    X, y_main, y_star, idxs = make_windows(encoded, targets_main, targets_star)
    dates = df[DATE_COL].iloc[idxs].reset_index(drop=True)

    # Check what dates have matching years so we can filter the test,val,trainig samples
    val_mask = year_mask(dates, VAL_YEARS)
    test_mask = year_mask(dates, TEST_YEARS)
    training_mask = ~(val_mask | test_mask)

    X_train, ym_train, ys_train = X[training_mask], y_main[training_mask], y_star[training_mask]

    X_val, ym_val, ys_val = X[val_mask], y_main[val_mask], y_star[val_mask]

    X_test, ym_test, ys_test = X[test_mask], y_main[test_mask], y_star[test_mask]

    print("Shapes:\n")
    # shape -> tells the dimensions of an array/matrix
    print("\nTrain:", X_train.shape, ym_train.shape, ys_train.shape)
    print("\nVal  :", X_val.shape,   ym_val.shape,   ys_val.shape)
    print("\nTest :", X_test.shape,  ym_test.shape,  ys_test.shape)

    model = build_recurrent_nn()

    cb = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]

    history = model.fit(
    X_train, {"main_probs": ym_train, "star_probs": ys_train},
    validation_data=(X_val, {"main_probs": ym_val, "star_probs": ys_val}),
    epochs=200, batch_size=64, callbacks=cb, verbose=1
    )

    numbers_array = []
    stars_array = []

    for i in range(100):
        mp, sp = model.predict(X_val[-1:], verbose=0)
        
        numbers, stars = sample_draw_probs(mp[0], sp[0])

        numbers_array.append(numbers)
        stars_array.append(stars)

    frequency_n = []
    frequency_s = []

    for i in range(N_MAINS + 1):
        frequency_n.append(0)

    for i in range(N_STARS + 1):
        frequency_s.append(0)

    for element_array in numbers_array:
        for i in element_array:
            frequency_n[i] += 1

    for element_array in stars_array:
        for i in element_array:
            frequency_s[i] +=1

    final_nums = []
    final_stars = []

    i = 1
    while i <= 5:
        max = 0
        pos = 0

        for x, value in enumerate(frequency_n):
            if value > max:
                max = value
                pos = x
    
        frequency_n[pos] = -1
        i +=1
        final_nums.append(pos)

    i = 1

    while i <= 2:
        max = 0
        pos = 0

        for x, value in enumerate(frequency_s):
            if value > max:
                max = value
                pos = x
        
        frequency_s[pos] = -1
        i += 1
        final_stars.append(pos)


    print("\nPredicted numbers:", final_nums)
    print("\nPredicted stars:", final_stars)

if __name__ == "__main__":
    main()