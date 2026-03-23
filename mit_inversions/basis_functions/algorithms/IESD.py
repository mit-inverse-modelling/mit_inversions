# IESD.py
# Created: 2 March 2026

import numpy as np 
from scipy.ndimage import gaussian_filter, sobel
from scipy.ndimage import label

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import disk
from skimage.filters import rank

from sklearn.cluster import KMeans

class IWASP:
    """
    Information-Weighted Adaptive Spatial Discretiation (IWASP) algorithm for
    creating basis function regions of footprint-flux fields. 

    Each region represents one unit of information content in
    the inverse problem where each unit is one unit of inferability.

    This algorithm is designed to be applied to point source emitters. 
    The algorithm works as follows:

    1. Calculate the entropy-weight from the footprint-flux field and its error.
    2. A watershed segmentation is applied to the entropy-weighted footprint-flux field to create regions of equal information content.
    3. The variability of each region is calculated based on the variance of the footprint-flux field and its error within the region.
    4. Regions with high variability are split using K-means clustering to create more homogeneous regions.
    5. The process is repeated until a target number of regions is reached or the variability threshold is met.
    
    use the .run() method to use the algorithm

    """
    
    def __init__(self, 
                 fp_flux_grid: np.ndarray, 
                 fp_flux_grid_error: np.ndarray, 
                 target_regions: int=80,
                 max_iter: int=1000,
                 var_threshold=None,
                 alpha: float=1.0, 
                 smooth_sigma: float=1.2,
                 ):
        """
        Initialize the IESD algorithm with the footprint-flux grid, 
        its error, and parameters for segmentation.

        Parameters:
        - fp_flux_grid (np.ndarray): 
            2D array of footprint-flux values.
        - fp_flux_grid_error (np.ndarray): 
            2D array of errors associated with the footprint-flux values.
        - target_regions (int): 
            Desired number of regions to segment into.
        - max_iter (int): 
            Maximum number of iterations for the adaptive partitioning.
        - var_threshold (int): 
            Variability threshold for splitting regions (if None, it will be set adaptively based on the distribution of variability).
        - alpha (float): 
            Weighting factor for the error in the composite field calculation.
        - smooth_sigma (float): 
            Sigma for Gaussian smoothing of the composite field before seed detection.
        """
        self.fp_flux_grid = fp_flux_grid
        self.fp_flux_grid_error = fp_flux_grid_error
        self.target_regions = target_regions
        self.max_iter = max_iter
        self.var_threshold = var_threshold
        self.alpha = alpha
        self.smooth_sigma = smooth_sigma


    def compute_composite_field(self):
        """
        Compute the composite field by weighting the footprint-flux field with its error.

        The weighting is done by multiplying the footprint-flux field with a factor 
        that increases with the error, controlled by the alpha parameter.
        """
        # Entropy weighted footprint-flux field
        G = self.fp_flux_grid * (1.0 + self.alpha * self.fp_flux_grid_error)

        # Smooth the composite field
        G_smooth = gaussian_filter(G, sigma=self.smooth_sigma)
        return G_smooth

    def detect_seeds(self, G, min_distance=2, threshold_rel=0.02):
        """
        Detect seed points for watershed segmentation using local maxima 
        in the composite field.

        Parameters:
        - G (np.ndarray):
            2D array composite field calculated from the footprint-flux field
        - min_distance (int):
            Specified minimum distance (in pixels) between neighbouring peaks.
            Any peak that has a taller neighbour within the distance is
            suppressed. 
        - threshold_rel (float):
            Value to filter peaks based on their relative intensity values
            ignoring peaks below a certain height.
        """
        coordinates = peak_local_max(G, 
                                     min_distance=min_distance, 
                                     threshold_rel=threshold_rel, 
                                     exclude_border=False)
        
        seed_mask = np.zeros_like(G, dtype=bool)
        for r, c in coordinates:
            seed_mask[r, c] = True
        markers, _ = label(seed_mask)
        return markers

    def initial_segmentation(self, G, markers):
        """
        Perform initial watershed segmentation using the detected 
        seed points.
        
        Parameters:
        - G (np.ndarray):
            2D array composite field calculated from the footprint-flux field
        - markers (np.ndarray):
            Array of values determined from peak_local_max
        """
        # Compute the distance transform of the composite field
        # We want to segment based on high values, so we take the negative
        labels = watershed(-G, markers=markers, connectivity=1)
        return labels

    def region_variability(self, G, H, labels, region_id, lam1=0.3, lam2=0.5):
        """
        Calculate the variability of a region based on the 
        variance of G and the mean of G and H within the region.

        Parameters:
        - G (np.ndarray):
            2D array composite field calculated from the footprint-flux field
        - H (np.ndarray):
            2D array footprint-flux error field 
        - labels ():

        - region_id ():

        - lam1 (float=0.3):
            Relative weighting of the mean of the masked composite field values
        - lam2 (float=0.5):
            Relative weighting of the mean of the masked uncertainty values
        """
        mask = labels == region_id
        
        Gv = G[mask]
        Hv = H[mask]
        if len(Gv)<5:
            return 0.0
    
        V = (np.var(Gv) + lam1 * np.mean(Gv) + lam2 * np.mean(Hv))
        return V

    def split_region(self, G, H, labels, region_id, n_splits=2):
        """
        Initiate K-means clustering algorithm for partioning composite field
        """
        mask = labels == region_id

        # Feature vectors 
        coords = np.column_stack(np.where(mask))
        values = np.column_stack((G[mask], H[mask]))

        km = KMeans(n_clusters=n_splits, n_init=75)
        cl = km.fit_predict(values)

        new_labels = labels.copy()
        max_label = labels.max()

        for i in range(n_splits):
            max_label +=1 
            idx = coords[cl == i]
            new_labels[idx[:,0], idx[:,1]] = max_label
        return new_labels

    def merge_regions(self, labels, region_list):
        """
        Merge each low-variability region with its most similar neighbor.
        """
        labels = labels.copy()

        for rid, _ in region_list:
            mask = labels == rid

            # find neighbors
            border = np.zeros_like(labels, dtype=bool)
            border[:-1, :] |= mask[1:, :]
            border[1:, :] |= mask[:-1, :]
            border[:, :-1] |= mask[:, 1:]
            border[:, 1:] |= mask[:, :-1]

            neighbors = np.unique(labels[border])
            neighbors = neighbors[(neighbors != rid) & (neighbors != 0)]

            if len(neighbors) == 0:
                continue

            # merge into smallest neighbor region (stability heuristic)
            neighbor_sizes = {n: np.sum(labels == n) for n in neighbors}
            target = min(neighbor_sizes, key=neighbor_sizes.get)

            labels[mask] = target

        return labels

    def adaptive_partition_convergent(self, max_splits_per_iter=3, max_merges_per_iter=3, tol=2):
        """
        Adaptive partitioning with convergence check based on the number of regions.
        """
        # Compute composite field
        G = self.compute_composite_field()
        
        # Detect seeds for watershed
        markers = self.detect_seeds(G)
        
        # Initial segmentation
        labels = self.initial_segmentation(G, markers)

        for it in range(self.max_iter):
            region_ids = np.unique(labels)
            region_ids = region_ids[region_ids != 0]
            n_regions = len(region_ids)

            # ---- Convergence check ----
            if abs(n_regions - self.target_regions) <= tol:
                print(f"Converged at iter {it}, regions={n_regions}")
                break

            # ---- Compute normalized variabilityv ----
            V = {}
            for rid in region_ids:
                mask = labels == rid
                area = np.sum(mask)

                if area < 5:
                    V[rid] = 0.0
                    continue

                Gv = G[mask]
                Hv = self.fp_flux_grid_error[mask]

                # Normalized variability (scale invariant)
                V[rid] = (np.var(Gv) / (np.mean(Gv)**2 + 1e-6) + 0.3*np.mean(Hv))

            # ---- Sort regions by variability
            sorted_regions = sorted(V.items(), key=lambda x: x[1], reverse=True)

            # ---- SPLIT CONTROL ---- 
            if n_regions < self.target_regions - tol:
                n_to_split = min(max_splits_per_iter, (self.target_regions - tol)-n_regions)

                # split highest variability regions
                for rid, _ in sorted_regions[:n_to_split]:
                    labels = self.split_region(G, self.fp_flux_grid_error, labels, rid, n_splits=2)

            #  ---- MERGE CONTROL ----
            elif n_regions > self.target_regions + tol:
                n_to_merge = min(max_merges_per_iter, n_regions - (self.target_regions + tol))
                
                # merge lowest variability regions 
                low_var_regions = sorted_regions[::-1][:n_to_merge]
                labels = self.merge_regions(labels, low_var_regions)

            # ---- STABILITY SAFETY ----
            else:
                print(f"Stabilized at iter {it}, regions={n_regions}")
                break

        return labels, G
    
    def run(self):
        """
        Run the algorithm!
        """
        labels_out, G_out = self.adaptive_partition_convergent()
        return labels_out, G_out