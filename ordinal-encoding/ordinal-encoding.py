def ordinal_encoding(values, ordering):
    mapping = {}

    for i, value in enumerate(ordering):
        mapping[value] = i

    return [mapping[value] for value in values]