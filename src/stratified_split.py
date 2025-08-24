# Usualy what we do is split data into test and training sets usually by picking random rows.
#
# Why is this harmful?
# There is a big chance (in this case) for the test data to only have benign data
#
# A stratified split ensures that the proportion of each train/test sample match the overall source of truth
#
# If the overall data has: 20% Malicious / 80% , it will make sure that the sample will have that percentage
#
import numpy as np

def stratified_split(x, y, test_size=0.2, val_size=0.2, seed=42):

    # To be trained data
    train_idxs, val_idxs, test_idxs = [], [], []

    #Random generator, by keep the seed with the same value you're ensuring that the
    # generated values are always the same
    rng = np.random.default_rng(seed=seed)

    y_len = len(y)

    # Create an array that counts from 0 to y_len, if y_len = 5 (which represents the n of rows in your data sample)
    # The array would be [0, 1, 2, 3, 4, 5]
    idxs = np.arange(y_len)

    # np.unique finds the unique values in an array in our use case: [0,1]
    # cls stands for class (in this case either class-0 / class-1)
    for cls in np.unique(y):
        matched_cls_idxs = idxs[y == cls]

        # We shuffle so we avoid for instance class-0 to be both in test and training samples
        rng.shuffle(matched_cls_idxs)

        n_cls = len(matched_cls_idxs)

        # for instance if the n_cls (the number of class-0s) is 50, we would do 50 * 0.2 = 10, then n_test = 10
        # np.floor rounds the number down to the nearest integer

        # Test data
        n_test = int(np.floor(test_size * n_cls))

        # Validation data
        n_val = int(np.floor(val_size * n_cls))

        # Training data
        # training will have 60% of the data
        n_training = n_cls - n_test - n_val

        # if matched_cls_idxs is [7, 2, 0, 9, 5, 3, 8, 1, 4, 6] and n_traing = 6
        # train_idxs: [7, 2, 0, 9, 5, 3]
        train_idxs.append(matched_cls_idxs[:n_training])

        # for val_idxs=2 we cut of the samples assigned to train_idxs and we would get: [8, 1]
        val_idxs.append(matched_cls_idxs[n_training:n_training+n_val])

        # the remaining
        test_idxs.append(matched_cls_idxs[n_training+n_val:])

    # In this use case, we will get a list of 2 arrays, one for each class
    # print(f"Train idxs: {train_idxs}\n\n")
    # print(f"Value idxs: {val_idxs}\n\n")
    # print(f"Testing idxs: {test_idxs}\n\n")

    final_train_idxs = np.concatenate(train_idxs)

    final_val_idxs = np.concatenate(val_idxs)

    final_test_idxs = np.concatenate(test_idxs)

    #print(f"Final Train idxs: {final_train_idxs}\n\n")
    #print(f"Final Value idxs: {final_val_idxs}\n\n")
    #print(f"Final Testing idxs: {final_test_idxs}\n\n")

    return (
        x[final_train_idxs], y[final_train_idxs],
        x[final_val_idxs], y[final_val_idxs],
        x[final_test_idxs], y[final_test_idxs]
        )


