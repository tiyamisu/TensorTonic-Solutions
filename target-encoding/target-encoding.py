def target_encoding(categories, targets):
    means = {}

    for category in set(categories):
        values = [targets[i] for i in range(len(categories)) if categories[i] == category]
        means[category] = sum(values) / len(values)

    return [means[category] for category in categories]