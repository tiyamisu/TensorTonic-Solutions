def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    dimensions = len(points[0])
    centroids = []
    for cluster in range(k):
        cluster_points = []
        for i in range(len(points)):
            if assignments[i] == cluster:
                cluster_points.append(points[i])
        if len(cluster_points) == 0:
            centroids.append([0.0] * dimensions)
        else:
            mean = []
            for d in range(dimensions):
                total = 0
                for point in cluster_points:
                    total += point[d]
                mean.append(total / len(cluster_points))
            centroids.append(mean)
    return centroids