import numpy as np
from function import FunctionFactory, Function
from manifold import Manifold, is_outer_or_inner, find_nearest_point
from grid import Grid

class Batch:
    dimentions : int
    border_normals : np.ndarray
    border_values : Function
    border_partials : list[Function]
    space_values : Function
    is_outer : np.ndarray
    is_inner : np.ndarray

    sample_volume : float
    phi_values : np.ndarray


class BatchGenerator:

    def __init__(
            self,
            dimentions : int,
            manifold : Manifold,
            grid : Grid,
            function_factory : FunctionFactory,
            distance_eps : float):
        self.__dimentions = dimentions
        self.__manifold = manifold
        self.__grid = grid
        self.__function_factory = function_factory
        self.__distance_eps = distance_eps
        self.__random_generator = np.random.default_rng(0)

    def create_batch(self, manifold_points : int, space_points : int)->Batch:
        batch = Batch()
        ff = self.__function_factory

        border_points, border_normals = self.__sample_manifold(manifold_points)
        grid_points, is_outer, is_inner = self.__sample_grid(space_points)

        batch.dimentions = self.__dimentions
        
        batch.border_normals = border_normals
        
        batch.border_values = ff.function(border_points)
        batch.border_partials = [
            ff.partial_derivative(border_points, i) for i in range(self.__dimentions)
        ]

        batch.space_values = ff.function(grid_points)
        batch.is_outer = is_outer
        batch.is_inner = is_inner
        """
        batch.sample_volume = self.__grid.sample_volume()

        nearest_point_index = find_nearest_point(grid_points, border_points)
        nearest_point = border_points[:, nearest_point_index]
        distance = np.sqrt(np.sum((grid_points-nearest_point)**2, axis=0))
        sdf_value = distance * is_outer - distance * is_inner
        batch.phi_values = -np.exp(-sdf_value) + 1

        print(sdf_value)
        """
        return batch

    
    def __sample_manifold(self, manifold_points : int)->tuple[np.ndarray, np.ndarray]:
        size = self.__manifold.points().shape[1]
        indicies = self.__indicies(size, manifold_points)
        return self.__manifold.points()[:, indicies], self.__manifold.normals()[:, indicies]
        #return self.__manifold.points(), self.__manifold.normals()

    def __sample_grid(self, grid_points : int)->tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = self.__grid.points_count()
        indicies = self.__indicies(size, grid_points)
        #indicies = np.arange(0, size)
        points = self.__grid.sample(indicies)
        is_outer, is_inner = is_outer_or_inner(points, self.__manifold, self.__distance_eps)
        return points, is_outer, is_inner

    def __indicies(self, array_size : int, count : int)->np.ndarray:
        return self.__random_generator.integers(0, array_size, count, dtype=np.uint64)
