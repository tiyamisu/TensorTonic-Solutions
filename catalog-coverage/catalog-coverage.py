def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """

    if n_items == 0:
        return 0.0

    unique_items = set()

    for recommendation_list in recommendations:
        unique_items.update(recommendation_list)

    return float(len(unique_items) / n_items)