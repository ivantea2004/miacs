import numpy as np


"""
    Stores information about function values in particular points
    Can be used to evaluate function at points with specified parameters
"""
class Function:
    
    """
        takes array of coefficients with indicies
        c[point_number][term_number]
    """
    def __init__(self, coefficients : np.ndarray):
        self.__coefficients = coefficients

    """
        Calculates function for every point with specified parameters
    """
    def eval(self, params : np.ndarray)->np.ndarray:
        return np.sum(self.__coefficients * params, axis=1)

    """
        Calculates function gradient by parameters at specified points
        Returns array with indicies
        return_array[parameter_index][point_index]
    """
    def eval_grad(self)->np.ndarray:
        return self.__coefficients.swapaxes(0, 1)

"""
    Is used to create function templates in given points
"""
class FunctionFactory:
    
    def __init__(self, dimentions, degree):
        self.__dimentions = dimentions
        self.__degree = degree
        self.__variable_factors = np.array([
            self.__variable_factor(
                self.__dimentions,
                self.__degree,
                i) for i in range(dimentions)
            ])
        
    
    def params_count(self)->int:
        return self.__args_count() * 2
    
    def __args_count(self)->int:
        return self.__degree ** self.__dimentions

    """
        Creates function templates in given points
    """
    def function(self, points : np.ndarray)->Function:
        args = self.__arguments(points)
        coss = np.cos(args)
        sins = np.sin(args)
        t = np.ndarray([points.shape[1], self.params_count()], dtype=np.float64)
        #t[:, 0] = 1
        t[:, 0:self.__args_count()] = coss
        t[:, self.__args_count():] = sins
        return Function(t)

    """
        Creates function partial derivative template by given variable
    """
    def partial_derivative(self, points : np.ndarray, variable_index)->Function:
        args = self.__arguments(points)
        coss = np.cos(args)
        sins = np.sin(args)
        t = np.ndarray([points.shape[1], self.params_count()], dtype=np.float64)
        #t[:, 0] = 0
        t[:, 0:self.__args_count()] = -sins * self.__variable_factors[variable_index]
        t[:, self.__args_count():] = coss * self.__variable_factors[variable_index]
        return Function(t)

    def __arguments(self, points : np.ndarray)->np.ndarray:
        #print(self.__variable_factors)
        return np.sum(np.array([
            np.outer(points[i], self.__variable_factors[i]) for i in range(self.__dimentions)
        ]), axis=0)

    @staticmethod
    def __variable_factor(dimentions, degree, variable_index)->np.ndarray:
        factors = np.arange(0, degree) # [1, 2, 3]
        md_factors = np.broadcast_to(factors, shape=[degree]*dimentions) # [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        md_factors_i = md_factors.swapaxes(variable_index, dimentions-1) # [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        return md_factors_i.reshape([degree**dimentions]) # [1, 1, 1, 2, 2, 2, 3, 3, 3]

    """
        Returns text representation of functin with given params
    """
    def render(self, params)->str:
        params = [np.format_float_positional(i) for i in params]
        args = self.__render_arguments()
        return '+'.join([
            f'{params[i]}cos({args[i]}) + {params[i + self.__args_count()]}sin({args[i]})' for i in range(self.__args_count())
        ])


    def __render_arguments(self)->list[str]:
        var_names = [chr(i + ord('x')) for i in range(3)]

        c = self.__args_count()
        return ['+'.join(f'{self.__variable_factors[j][i]}{var_names[j]}' for j in range(self.__dimentions)) for i in range(c)]