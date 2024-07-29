import numpy as np


def gaussian_mixture_batch(batch_size, tsp_size, cdist):
    # Generate from Gaussian mixture, taken from Zhang et. al (2022)
    points = []
    for i in range(batch_size):
        nc = np.random.randint(3, 7)
        nums = np.random.multinomial(tsp_size, np.ones(nc)/nc)
        xy = []
        for num in nums:
            center = np.random.uniform(0, cdist, size=(1,2))
            nxy = np.random.multivariate_normal(
                mean=center.squeeze(),
                cov=np.eye(2,2),
                size=(num,)
            )
            xy.extend(nxy)
        points.append(np.array(xy))
    points = np.array(points)
    
    # Translate and scale points to fit inside unit square
    points -= np.min(points, axis=1, keepdims=True)
    points /= np.max(
        np.max(
            points, axis=1, keepdims=True
        ), axis=2, keepdims=True
    )
    
    return points
