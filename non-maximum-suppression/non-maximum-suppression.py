def nms(boxes, scores, iou_threshold):
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    kept = []
    while order:
        current = order.pop(0)
        kept.append(current)
        remaining = []
        for index in order:
            left = max(boxes[current][0], boxes[index][0])
            top = max(boxes[current][1], boxes[index][1])
            right = min(boxes[current][2], boxes[index][2])
            bottom = min(boxes[current][3], boxes[index][3])
            intersection = max(0, right - left) * max(0, bottom - top)
            area_current = (boxes[current][2] - boxes[current][0]) * (boxes[current][3] - boxes[current][1])
            area_other = (boxes[index][2] - boxes[index][0]) * (boxes[index][3] - boxes[index][1])
            union = area_current + area_other - intersection
            overlap = intersection / union if union else 0.0
            if overlap < iou_threshold:
                remaining.append(index)
        order = remaining
    return kept
