import numpy as np

from constants import MAIN_COLS, STAR_COLS, N_MAINS,D, N_STARS,T

def encode_draw(row):

    # Converts into an array of 62 positions of 0s
    v = np.zeros((D,), dtype=np.float32)

    # Based on the row it will have the 1 value for a nomral number and a star
    for c in MAIN_COLS:
        v[row[c] - 1] = 1.0

    for c in STAR_COLS:
        v[N_MAINS + (row[c] - 1)] = 1.0

    return v

def draw_to_freq_targets(row):

    # Make a star and a num vec with only 0
    main_t = np.zeros((N_MAINS,), dtype=np.float32)
    star_t = np.zeros((N_STARS,), dtype=np.float32)

    # MAIN COLS -> has for example 3,45,30 
    # 
    for c in MAIN_COLS:
        main_t[row[c] - 1] += 1.0
    for c in STAR_COLS:
        star_t[row[c] - 1] += 1.0

    # Convert into percentages
    main_t /= main_t.sum()
    star_t /= star_t.sum()

    return main_t, star_t

def make_windows(X_all, y_main_all, y_star_all):
    X_list, ym_list, ys_list, idx_list = [], [], [], []

    for i in range(T, len(X_all)):

        # for every i+1, it's added a list of 100 position (the current plus the previous 100)
        X_list.append(X_all[i-T:i])

        # A vector of the percentages
        ym_list.append(y_main_all[i])
        ys_list.append(y_star_all[i])
        idx_list.append(i)

    return (np.stack(X_list), np.stack(ym_list), np.stack(ys_list), np.array(idx_list))

def year_mask(dates, years):
    return dates.dt.year.isin(years).to_numpy()