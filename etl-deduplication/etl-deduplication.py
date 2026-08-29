def deduplicate(records: list, key_columns: list, strategy: str) -> list:
    """
    Returns a list of selected records.
    """
    groups = {}
    key_order = []

    for index, record in enumerate(records):
        key = tuple(record[column] for column in key_columns)

        if key not in groups:
            groups[key] = []
            key_order.append(key)

        groups[key].append((index, record))

    result = []

    for key in key_order:
        group = groups[key]

        if strategy == "first":
            selected = group[0][1]

        elif strategy == "last":
            selected = group[-1][1]

        elif strategy == "most_complete":
            # min() is stable, so ties retain the first occurrence.
            selected = min(
                group,
                key=lambda item: sum(
                    value is None
                    for value in item[1].values()
                )
            )[1]

        else:
            # Valid inputs should only use the three specified strategies.
            selected = group[0][1]

        result.append(selected)

    return result