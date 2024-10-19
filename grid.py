import numpy as np

class Grid:
    
    """
        bounds = [
            [x_0, y_0],
            [x_1, y_1]
        ]
    """
    def __init__(self, dimentions : int, bounds :  np.ndarray, density : int):
        self.__dimentions = dimentions
        self.__bounds = bounds
        self.__density = density
    
    def volume(self)->float:
        return np.prod(self.__bounds[1]-self.__bounds[0])
    def sample_volume(self)->float:
        return self.volume() / self.points_count()

    def points_count(self)->int:
        return self.__density ** self.__dimentions

    """
        Returns array of points with specified indices
        [
            [x_0, x_1, x_2],
            [y_0, y_1, y_2]
        ]    
    """
    def sample(self, indicies : np.ndarray)->np.ndarray:
        md_indicies = np.unravel_index(indicies, shape=[self.__density]*self.__dimentions)
        int_grid_points = np.array(md_indicies, dtype=np.float64)
        unit_grid_points = int_grid_points / (self.__density - 1)
        return unit_grid_points * (self.__bounds[1]-self.__bounds[0]) + self.__bounds[0]