def validate_records(records: list, schema: list) -> list:
    """
    Returns a list of result dictionaries.
    """
    results = []

    for record_index, record in enumerate(records):
        errors = []

        for entry in schema:
            column = entry["column"]

            # 1. Missing value
            if column not in record:
                errors.append(f"{column}: missing")
                continue

            value = record[column]

            # 2. Nullability
            if value is None:
                if entry["nullable"]:
                    continue
                else:
                    errors.append(f"{column}: null")
                    continue

            expected_type = entry["type"]

            # 3. Type checking
            if expected_type == "int":
                valid_type = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                )
                actual_type = type(value).__name__

            elif expected_type == "float":
                valid_type = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                )

                # An int is accepted as a float according to the problem.
                actual_type = (
                    "float"
                    if valid_type
                    else type(value).__name__
                )

            elif expected_type == "str":
                valid_type = isinstance(value, str)
                actual_type = type(value).__name__

            else:
                valid_type = False
                actual_type = type(value).__name__

            if not valid_type:
                errors.append(
                    f"{column}: expected {expected_type}, got {actual_type}"
                )
                continue

            # 4. Inclusive range checking
            if "min" in entry and value < entry["min"]:
                errors.append(f"{column}: out of range")
                continue

            if "max" in entry and value > entry["max"]:
                errors.append(f"{column}: out of range")

        results.append({
            "record_index": record_index,
            "is_valid": len(errors) == 0,
            "errors": errors
        })

    return results