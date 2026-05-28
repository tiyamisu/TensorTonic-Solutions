def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """

    assignments = []

    for point in points:

        min_dist = float('inf')
        cluster_index = -1

        for i in range(len(centroids)):

            dist = 0

            for j in range(len(point)):
                dist += (point[j] - centroids[i][j]) ** 2

            if dist < min_dist:
                min_dist = dist
                cluster_index = i

        assignments.append(cluster_index)

    return assignments