def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """

    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # Intersection coordinates
    x1 = max(x1_a, x1_b)
    y1 = max(y1_a, y1_b)
    x2 = min(x2_a, x2_b)
    y2 = min(y2_a, y2_b)

    # Intersection area
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    # Individual areas
    area_a = max(0, x2_a - x1_a) * max(0, y2_a - y1_a)
    area_b = max(0, x2_b - x1_b) * max(0, y2_b - y1_b)

    # Union
    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return float(intersection_area / union_area)