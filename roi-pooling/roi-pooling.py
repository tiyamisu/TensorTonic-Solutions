def roi_pool(feature_map: list, rois: list, output_size: int) -> list:
    """
    Returns a list of pooled grids.
    """
    H = len(feature_map)
    W = len(feature_map[0])

    S = output_size

    results = []

    for roi in rois:
        x1, y1, x2, y2 = roi

        roi_h = y2 - y1
        roi_w = x2 - x1

        pooled = []

        for i in range(S):
            row = []

            h_start = y1 + (i * roi_h) // S
            h_end = y1 + ((i + 1) * roi_h) // S

            # Expand an empty bin to one pixel.
            if h_start == h_end:
                h_end = min(h_start + 1, y2)

            for j in range(S):
                w_start = x1 + (j * roi_w) // S
                w_end = x1 + ((j + 1) * roi_w) // S

                # Expand an empty bin to one pixel.
                if w_start == w_end:
                    w_end = min(w_start + 1, x2)

                values = []

                for r in range(h_start, h_end):
                    for c in range(w_start, w_end):
                        if 0 <= r < H and 0 <= c < W:
                            values.append(feature_map[r][c])

                # The ROI is assumed to be valid and non-empty.
                row.append(max(values))

            pooled.append(row)

        results.append(pooled)

    return results