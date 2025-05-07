from concurrent.futures import ThreadPoolExecutor
from math import log
from scipy.interpolate import RegularGridInterpolator, interp1d
import numpy as np


class ParallelBase:
    """Base class for parallelized function evaluation.
    
    Features:
    - Automatic thread pool management
    - Context manager support for resource cleanup
    - Batch evaluation of multiple points
    - Graceful error handling
    
    Parameters
    ----------
    max_workers : int, optional
        Maximum parallel threads; None uses the default setup in ThreadPoolExecutor (default: None)
    """
    
    def __init__(self, max_workers=None):
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        
    def __del__(self):
        """Safely shutdown thread pool when instance is destroyed."""
        if self.pool._threads:
            self.pool.shutdown(wait=True)

    def __enter__(self):
        """Enable context manager usage ('with') for resource control."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically clean up resources when exiting context."""
        self.pool.shutdown(wait=True)
        return False

    def reset_pool(self, max_workers=None):
        """Reinitialize thread pool with new worker count.
        
        Args:
            max_workers : int, optional
                New maximum parallel threads (default: auto-scale)
        """
        self.pool.shutdown(wait=True)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def _parallel_execute(self, func, points):
        """Execute function in parallel over points.
        
        Args:
            func: Function to execute
            points: Iterable of points to process
            
        Returns:
            List of results
            
        Raises:
            RuntimeError: If execution fails
        """
        try:
            futures = self.pool.map(func, points)
            return list(futures)
        except Exception as e:
            self.pool.shutdown(wait=False)
            raise RuntimeError(f"Parallel execution failed: {str(e)}") from e


class ParallelInterpolator(ParallelBase):
    """Parallelized interpolator for regular grid data.
    
    Parameters
    ----------
    data_grid : ndarray
        N-dimensional array of data values
    grid_points : tuple of arrays
        Tuple specifying grid coordinates for each dimension
    max_workers : int, optional
        Maximum parallel threads (default: None)
    
    Attributes
    ----------
    interp : RegularGridInterpolator
        Underlying SciPy interpolator instance

    Example
    -------
    >>> import numpy as np
    >>> grid = np.linspace(0, 1, 10)
    >>> data = np.sin(grid)
    >>> interpolator = ParallelInterpolator(data, (grid,))
    >>> interpolator([0.2, 0.5, 0.8])
    >>> del interp
    >>> # or
    >>> with ParallelInterpolator(data, (grid,), max_workers=30) as interpolator:
    >>>     points = np.array([[0.2], [0.5], [0.8]])  # Must be 2D array for 1D grid
    >>>     results = interpolator(points)
    >>>     print("Interpolated values:", results)
    """
    
    def __init__(self, data_grid, grid_points, max_workers=None):
        super().__init__(max_workers)
        self.interp = RegularGridInterpolator(grid_points, data_grid)
        
    def __call__(self, points):
        """Evaluate interpolator at multiple points in parallel.
        
        Args:
            points : array_like
                (N, D) array of N points with D dimensions. For example, 1D grid: [[x1], [x2], ...].
                
        Returns:
            ndarray: Interpolated values at input points
        """
        return self._parallel_execute(self.interp, points)


    
# def loglog_interp_func_1d(input_x_grid, input_y_grid, kind='linear', log_x_input=False, log_y_input=True):
#     # Transform to log space if input is linear
#     x_grid = input_x_grid.copy()
#     if not log_x_input:
#         x_grid = np.log10(x_grid)
    
#     y_grid = input_y_grid.copy()
#     if not log_y_input:
#         y_grid = np.log10(y_grid)
        
#     # Verify monotonicity after transformations
#     if not np.all(np.diff(x_grid) > 0):
#         raise ValueError("Grid points must be monotonically increasing in log space")
        
#     log_interp = interp1d(x_grid, y_grid, kind=kind, fill_value='extrapolate')
#     # Return function that handles log-transformations automatically
#     def wrapped_interp(xs):
#         return 10**log_interp(np.log10(xs))
    
#     return wrapped_interp

class Interpolator1D:
    def __init__(self, interp, logx=True, logy=True):
        '''
        Args:
            interp: 1D interpolator
            logx: whether the input (x) of interp is log scale
            logy: whether the output (y) of interp is log scale

        Returns:
            Interpolator1D function instance,
            whose input and output are both in ordinary scale.
        '''
        self.interp = interp
        self.logx = logx
        self.logy = logy
        # if logx and logy:
        #     self.__call__ = self.loglog
        # elif logx:
        #     self.__call__ = self.logx_uniformy
        # elif logy:
        #     self.__call__ = self.uniformx_logy
        # else:
        #     self.__call__ = self.uniformx_uniformy
        
    def loglog(self, xs):
        log_xs = np.log10(xs)
        log_ys = self.interp(log_xs)
        return 10.**log_ys

    def logx_uniformy(self, xs):
        log_xs = np.log10(xs)
        ys = self.interp(log_xs)
        return ys

    def uniformx_logy(self, xs):
        log_ys = self.interp(xs)
        return 10.**log_ys

    def uniformx_uniformy(self, xs):
        ys = self.interp(xs)
        return ys

    def __call__(self, xs):
        if self.logx and self.logy:
            return self.loglog(xs)
        elif self.logx:
            return self.logx_uniformy(xs)
        elif self.logy:
            return self.uniformx_logy(xs)
        else:
            return self.uniformx_uniformy(xs)


def interp_func_1d(log_x_grid, log_y_grid, kind='cubic'):
    '''
    Args:
        log_x_grid: 1D array, the log10 transformed coordinate grid feeded to the interpolator
        log_y_grid: 1D array, the log10 transformed function values
        kind: str, the interpolation method, default is 'cubic'
    Returns:
        Interpolator1D function instance, whose input and output are both in ordinary scale.
    '''
            
    # Verify monotonicity after transformations
    if not np.all(np.diff(log_x_grid) > 0):
        raise ValueError("Grid points must be monotonically increasing in log space")
        
    log_interp = interp1d(log_x_grid, log_y_grid, kind=kind, fill_value='extrapolate')
    
    return Interpolator1D(log_interp, logx=True, logy=True)


def homogeneous_dist(*args, **kwargs):
    """Constant distribution function that always returns 1"""
    return 1.0