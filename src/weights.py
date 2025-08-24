import numpy as np

def compute_class_weights(y):

    # In this case 2
    classes = np.unique(y)
    
    # This will return an array that each element is the number of times c matched with y per class
    counts = np.array([(y == c).sum() for c in classes], dtype=np.float32)

    # since counts is an array created by np, even if len(y) and len(classes) are 1 pure value numpy will apply broadcasting:
    # counts will iterate and weights will be filled one by one creating an array as well without having to specify
    weights = (len(y) / (len(classes) * counts))

    #A Class with High Weight is a rare class (doesn't appear that much on the data)
    #A Class with Low Weight is a common class (very predominant on the data)
    
    # The point of this is to boost the influence of rarer classes. Otherwise the model would just ignore them.

    # Zip will associate classes to its respective waits
    # For instance:
    # (0, 6)
    # (1, 10)
    return {int(c): float(w) for c, w in zip(classes, weights)}
