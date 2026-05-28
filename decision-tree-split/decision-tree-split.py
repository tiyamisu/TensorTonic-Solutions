import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    
    def gini(labels):
        total = len(labels)
        counts = {}
        
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        
        impurity = 1.0
        
        for count in counts.values():
            p = count / total
            impurity -= p * p
        
        return impurity

    n_samples = len(X)
    n_features = len(X[0])

    parent_gini = gini(y)

    best_gain = -1
    best_feature = -1
    best_threshold = -1

    for feature in range(n_features):

        values = sorted(set(row[feature] for row in X))

        for i in range(len(values) - 1):

            threshold = (values[i] + values[i + 1]) / 2

            left_y = []
            right_y = []

            for j in range(n_samples):
                if X[j][feature] <= threshold:
                    left_y.append(y[j])
                else:
                    right_y.append(y[j])

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            left_gini = gini(left_y)
            right_gini = gini(right_y)

            weighted_gini = (
                (len(left_y) / n_samples) * left_gini +
                (len(right_y) / n_samples) * right_gini
            )

            gain = parent_gini - weighted_gini

            if (gain > best_gain or
               (gain == best_gain and feature < best_feature) or
               (gain == best_gain and feature == best_feature and threshold < best_threshold)):

                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return [best_feature, best_threshold]