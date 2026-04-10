# RegionalSum.py
# Created: 2 March 2026

import numpy as np

class RegionalSum:
    """
    Regional sum algorithm for creating basis function regions
    from footprint-flux fields.

    Use the .run() method for calculating basis function regions.
    """

    def __init__(self,
                 fp_flux_grid: np.ndarray,
                 target_regions: int=40,
                 max_iter: int=1000,
                 ):
        """
        Initialize the RegionalSum algorithm with the footprint-flux grid 
        and parameters for segmentation.

        Parameters:
        - fp_flux_grid (np.ndarray): 
            2D array of footprint-flux values.
        - target_regions (int):
            Number of basis function regions to optimize for in the algorithm
        - max_iter (int):
            Maximum number of iterations to run the algorithm
        """

        self.fpXflux_grid = fp_flux_grid
        self.target_regions = target_regions
        self.max_iter = max_iter


    def bucket_value_split(self, grid, bucket, offset_x=0, offset_y=0, depth=0):
        """
        Binary geometric splitting with alternating X/Y directions.
        Split is always by midpoint of geometry, not values.

        Parameters
        - grid (np.ndarray):
            2D array for partitioning
        - bucket (float):
            Pre-calculated value for optimizing the number of
            basis function partitions to the target regions
        - offset_x (int):
            Value to offset the grid in the X-direction
        - offset_y (int):
            Value to offset the grid in the Y-direction
        - depth (int):
            Value used to determine geometric direction
            for basis function split. 
        """

        # Stop condition
        if np.sum(grid) <= bucket or grid.shape == (1,1):
            return [(offset_y, offset_y + grid.shape[0],
                     offset_x, offset_x + grid.shape[1])]

        # Alternate axis: even depth -> X, odd depth -> Y
        if depth % 2 == 0:
            # ---- Split in X (columns) ----
            half_x = grid.shape[1] // 2
            if half_x == 0:  # safety
                return [(offset_y, offset_y + grid.shape[0],
                         offset_x, offset_x + grid.shape[1])]

            left = grid[:, :half_x]
            right = grid[:, half_x:]

            return (
                self.bucket_value_split(left, bucket, offset_x, offset_y, depth+1) +
                self.bucket_value_split(right, bucket, offset_x+half_x, offset_y, depth+1)
            )

        else:
            # ---- Split in Y (rows) ----
            half_y = grid.shape[0] // 2
            if half_y == 0:  # safety
                return [(offset_y, offset_y + grid.shape[0],
                         offset_x, offset_x + grid.shape[1])]

            top = grid[:half_y, :]
            bottom = grid[half_y:, :]

            return (
                self.bucket_value_split(top, bucket, offset_x, offset_y, depth+1) +
                self.bucket_value_split(bottom, bucket, offset_x, offset_y+half_y, depth+1)
            )

    def get_nregions(self, bucket, grid):
        """
        Returns number of regions for bucket value

        Parameters:
        - bucket (float):
            Pre-calculated value for optimizing the number of
            basis function partitions to the target regions
        - grid (np.ndarray):
            2D array for partitioning           
        """
        regions = self.bucket_value_split(grid, bucket)
        return len(regions)

    def optimize_nregions(self, bucket, grid, tol):
        """
        Optimize bucket value to obtain nregion basis functions 
        with +/- tol

        Parameters:
        - bucket (float):
            Pre-calculated value for optimizing the number of
            basis function partitions to the target regions
        - grid (np.ndarray):
            2D array for partitioning
        - tol (int):
            Tolerance value for number of basis function regions
            to compute
        """
        current_bucket = bucket
        current_tol = tol 

        # Outer loop increases tol by +1 if convergence not achieved 
        # in max_iter range
        for _ in range(10):
            for j in range(self.max_iter):
                current_nregion = self.get_nregions(current_bucket, grid)

                if (current_nregion <= self.target_regions + current_tol) and \
                   (current_nregion >= self.target_regions - current_tol):

                    print(f"Optimal bucket value ({current_bucket}) after {j} iterations "
                          f"with tolerance {current_tol}")
                    return current_bucket
                
                # If too many regions -> bucket too small -> increase bucket
                if current_nregion > self.target_regions + current_tol:
                    # current_bucket *= 1.01
                    current_bucket *= 1.05
                # If too few regions -> bucket too large -> decrease bucket
                else:
                    # current_bucket *= 0.99
                    current_bucket *= 0.95
            current_tol += 1

        raise BufferError(f"Failed to converge for all tolerances from {tol} to {current_tol}!")

    def run(self, tol=2):
        """
        Run the algorithm!

        Parameters:
        - tol (int):
            Initial tolerance level for number of basis 
            function regions. 
        """
        # Use mean value of the footprint-flux grid as starting bucket value
        starting_value = np.nanmean(self.fpXflux_grid)
        
        # Optimize the bucket value for number of basis functions
        optimal_bucket = self.optimize_nregions(
            bucket=starting_value,
            grid=self.fpXflux_grid,
            tol=tol
        )
        
        # Calculate regions 
        regions = self.bucket_value_split(self.fpXflux_grid, optimal_bucket)
        
        # Convert region vertices to masked areas
        bf_grid = np.zeros_like(self.fpXflux_grid)
        for i in range(len(regions)):
            x_start, x_stop = regions[i][0], regions[i][1]
            y_start, y_stop = regions[i][2], regions[i][3]
            bf_grid[x_start:x_stop, y_start:y_stop] = i

        return optimal_bucket, bf_grid