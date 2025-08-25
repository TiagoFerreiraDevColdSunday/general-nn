import keras
import numpy as np
import pandas as pd

from stratified_split import stratified_split
from weights import compute_class_weights
from models import build_lstm
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def to_seq(x):
    # x.shape[0] -> nº of rows in a dataset (samples)
    # x.shape[1] -> nº of collumns in a dataset (features)

    # Returns an array that each element is the row of the data. Best format for LSTMs
    return x.reshape((x.shape[0], x.shape[1], 1))

# Data is from kaggle
#
def main():
    df = pd.read_csv("../data/cancer_data.csv")

    print("Columns:", df.columns.to_list())

    # Target Column (the value that we want to predict)
    candidate = ["diagnosis"]
    # Drop Columns that are no longer needed (eg. id collumns)
    drop_id_column = "id"

    # Get the values from the columns we want to predict
    target_col = next((c for c in candidate if c in df.columns), None)

    # Get the ID columns
    drops_col = [c for c in df.columns if any (x in c for x in drop_id_column)]

    # Updating by dropping the target collumn and the ids
    x_df = df.drop(columns=[target_col] + drops_col)

    y_raw = df[target_col]

    # Transform the targets (which currently are string) into unique numeric values
    classes, y = np.unique(y_raw.astype(str), return_inverse=True)
    print("\nLabel mapping:", {cls: i for i, cls in enumerate(classes)})

    print(f"\nTarget data classified: {y}")


    #
    #age  height
    #21    1.75
    #35    1.82
    #
    #Converts into: 
    # array([[21.  , 1.75],
    #        [35.  , 1.82]])
    #
    x = x_df.to_numpy().astype(np.float32)

    x_train, y_train, x_val, y_value, x_test, y_test = stratified_split(x, y)

    print(f"\nX_TRAIN: \n{x_train}")

    eps = 1e-7

    # mean will be a vector in which each element represents the mean of each column (using keepdims=True)
    mean = x_train.mean(axis=0, keepdims=True)


    # Deviation is how far the data is from the mean, the larger the std the more distant the data is from the mean
    # How is standard deviation calculated?:
    # if mean = 5 
    # For each column the std happens, in the first column let's suppose that: 
    # x = [2,3,4,5]
    # The sum would be: (2 - 5)^2 + (3 - 5)^2 + (4 - 5)^2 + (5 + 5)^2 = 9 + 4 + 1 = 14
    #
    # Then sqrt(14 / 4).  (sqrt (final_sum / n_elements_of_the_column) )
    std = x_train.std(axis=0, keepdims=True) + eps


    # We normalize.
    # Imagine: height and yearly_salary as data.
    # yearly_salary would probably explode the values and overshadow the height.
    # 
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    n_features = x_train.shape[1]

    # Calculate Weights
    weights = compute_class_weights(y)

    print(f"Calculated weights:\n{weights}")

    # If you used the data of the repo the weights are the following:
    # {0: 0.7969187498092651, 1: 1.3419811725616455}
    # Which means Benign Cancer is more common than Malignant Cancer


    # Early Stopping watches a metric and if i
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=10, # Even if bad results it won't just stop, it will wait until 10 bad epochs
        restore_best_weights=True # After stopping the training will reset to the best weights.
    )
    #
    # Epoch -> one full training through the dataset
    # AUC metric is between 0 and 1, if 0.5 the model is just flipping a coin, if 0.5<x<0.1, it's better than random.

    print("\n\n Training")

    x_train_seq = to_seq(x_train)
    x_val_seq = to_seq(x_val)
    x_test_seq = to_seq(x_test)

    lst_model = build_lstm(n_features)

    lst_model.fit(
        x_train_seq, y_train,
        validation_data=(x_val_seq, y_value),
        epochs=200, batch_size=32,
        class_weight=weights,
        verbose=0
    )

    predict = lst_model.predict(x_test_seq).ravel()

    pred_lstm = (predict >= 0.5).astype(int)
    print("\n=== LSTM (tabular-as-seq) ===")
    print("AUC:", roc_auc_score(y_test, pred_lstm))
    print(classification_report(y_test, pred_lstm, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred_lstm))

if __name__ == "__main__":
    print(main())
