import numpy as np

"""
    Class for manifold description
"""
class Manifold:
    def dimentions(self)->int:
        pass
    def points(self)->np.ndarray:
        pass
    def normals(self)->np.ndarray:
        pass

"""
    For every point in points1 find nearest point in points2 and returns
    array of theirs indicies
"""
def find_nearest_point(points1 : np.ndarray, points2 : np.ndarray)->np.ndarray:
    
    MAX_MATRIX_SIZE = 1000000
    MAX_SMALL_SIZE = max([MAX_MATRIX_SIZE / points1.shape[1], 1])
    SPLIT_COUNT = 10

    if True:#points2.shape[1] <= MAX_SMALL_SIZE:
        differences = np.array([
            np.outer(points1[i], np.ones(len(points2[i]), dtype=np.float64)) -
            np.outer(np.ones(len(points1[i]), dtype=np.float64), points2[i]) 
            for i in range(0, len(points1))
        ])
        distances = np.sum(differences**2, axis=0)
        return np.argmin(distances, axis=1)

    else:
        size = points2.shape[1]
        new_size = size // SPLIT_COUNT
        rem = size % SPLIT_COUNT
        ranges = [
            (i * new_size, (i+1)*new_size) for i in range(SPLIT_COUNT)
        ]
        if rem > 0:
            ranges.append((new_size * SPLIT_COUNT, size))
        results = [
            find_nearest_point(points1, points2[:, begin:end]) + begin for begin, end in ranges
        ]
        new_points2 = points2[:, results]
        new_results = find_nearest_point(points1, new_points2)
        return results[new_results]

"""
    Returns two bool arrays.
    The first for "is_outer" flag, second is for "is_inside" flag
"""
def is_outer_or_inner(points : np.ndarray, manifold : Manifold, distance_eps : float):
    nearest_index = find_nearest_point(points, manifold.points())
    nearest_point = manifold.points()[:, nearest_index]
    nearest_normal = manifold.normals()[:, nearest_index]
    nearest_difference = points-nearest_point
    nearest_distance = np.sqrt(np.sum(nearest_difference**2, axis=0))
    nearest_dot = np.sum(nearest_normal * nearest_difference, axis=0)
    is_near_border = (nearest_distance < distance_eps) * 1
    is_outer = (1-is_near_border) * (nearest_dot > 0)
    is_inner = (1-is_near_border) * (nearest_dot < 0)
    return is_outer, is_inner


class DiscreteManifold:

    def __init__(self, dims, p, n):
        self.__dims = dims
        self.__p = p
        self.__n = n
    def dimentions(self)->int:
        return self.__dims
    def points(self)->np.ndarray:
        return self.__p
    def normals(self)->np.ndarray:
        return self.__n

class Curve(Manifold):

    def __init__(self, x, y, dx, dy, t_bounds, samples):
        t = np.linspace(t_bounds[0], t_bounds[1], samples)
        self.__x = x(t)
        self.__y = y(t)
        dx_ = dx(t)
        dy_ = dy(t)
        self.__nx = dy_ / (dx_**2 + dy_**2)
        self.__ny = -dx_ / (dx_**2 + dy_**2)

    def dimentions(self):
        return 2
    def points(self):
        return np.array([self.__x, self.__y])
    
    def normals(self):
        return np.array([self.__nx, self.__ny])

class Surface(Manifold):

    def __init__(self, variables, partials, bounds, samples):
        self.__dims = len(variables)
        params0 = np.array(
            [np.linspace(bounds[i][0], bounds[i][1], samples) for i in range(self.__dims-1)]
        )

        def discrates_product(a, b):
            return np.transpose([np.tile(a, len(b)), np.repeat(b, len(a))]).swapaxes(0, 1)
        params = discrates_product(params0[0], params0[1])

        self.__coords = np.array([variables[i](params[0], params[1]) for i in range(self.__dims)])
        partials = np.array([
            [ partials[j][i](params[0], params[1]) for j in range(self.__dims)] for i in range(self.__dims-1)
        ]).swapaxes(1, 2)
        
        normals = np.cross(partials[0], partials[1])
        #print(partials)
        #print(normals)
        #exit()
        self.__normals = normals.swapaxes(0, 1) / np.sqrt(np.sum(normals**2, axis=1))

    def dimentions(self):
        return 3
    def points(self):
        return self.__coords
    def normals(self):
        return self.__normals