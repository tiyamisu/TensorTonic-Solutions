import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=float)
    classes = np.unique(y_train)
    means = {}
    variances = {}
    priors = {}
    for c in classes:
        X_c = X_train[y_train == c]
        means[c] = np.mean(X_c, axis=0)
        variances[c] = np.var(X_c, axis=0) + 1e-9
        priors[c] = len(X_c) / len(X_train)
    predictions = []
    for x in X_test:
        posteriors = []
        for c in classes:
            mean = means[c]
            var = variances[c]
            log_prior = np.log(priors[c])
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)
        predictions.append(classes[np.argmax(posteriors)])
    return predictions