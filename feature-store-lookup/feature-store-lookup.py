def feature_store_lookup(feature_store: dict, requests: list, defaults: dict) -> list:
    """
    Returns a list of feature dictionaries.
    """
    result = []

    for request in requests:
        user_id = request["user_id"]

        # Copy so we never modify the original feature store/defaults.
        features = dict(feature_store.get(user_id, defaults))

        # Add request-time online features.
        features.update(request["online_features"])

        result.append(features)

    return result