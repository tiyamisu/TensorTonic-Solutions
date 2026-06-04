import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    classes = np.unique(y_train)
    result = []

    for x in X_test:
        scores = []
        for c in classes:
            Xc = X_train[y_train == c]

            prior = np.log(len(Xc) / len(X_train))
            theta = (Xc.sum(axis=0) + 1) / (len(Xc) + 2)
            loglik = np.sum(x * np.log(theta) + (1 - x) * np.log(1 - theta))

            scores.append(prior + loglik)
            
        result.append(scores)
        
    return np.array(result)