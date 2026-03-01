import numpy as np


def aggregate_features(feature_list):

    if len(feature_list) == 0:
        raise ValueError("No facial data captured.")

    keys = feature_list[0].keys()

    avg_features = {}

    for k in keys:
        avg_features[k] = float(
            np.mean([f[k] for f in feature_list])
        )

    return avg_features