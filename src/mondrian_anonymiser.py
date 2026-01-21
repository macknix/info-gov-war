"""
Mondrian k-anonymity wrapper for NumPy arrays.

This module provides a clean interface to apply Mondrian multidimensional 
k-anonymity to continuous/discretised data, returning anonymised data that
can be used in downstream modelling tasks.

The Mondrian algorithm recursively partitions data along the dimension with
the largest normalised range, ensuring each partition has at least k records.
Records within a partition are generalised to the same value (e.g., bin midpoint).
"""

import numpy as np
from collections import Counter
from functools import cmp_to_key
import time
from typing import Tuple, List, Dict, Optional, Union


class MondrianPartition:
    """
    Represents a partition (equivalence class) in the Mondrian algorithm.
    
    Attributes:
        member: List of record indices in this partition
        low: Lower bounds for each quasi-identifier dimension
        high: Upper bounds for each quasi-identifier dimension
        allow: Boolean flags indicating if splitting is allowed on each dimension
    """
    
    def __init__(self, member_indices: List[int], low: List[float], high: List[float], n_dims: int):
        self.member = list(member_indices)
        self.low = list(low)
        self.high = list(high)
        self.allow = [True] * n_dims
    
    def __len__(self):
        return len(self.member)


class MondrianAnonymiser:
    """
    Mondrian k-anonymity implementation for numerical data.
    
    This implementation works directly with NumPy arrays and supports:
    - Strict Mondrian: partitions have no overlap
    - Relaxed Mondrian: allows some overlap to achieve better balance
    
    Parameters
    ----------
    k : int
        The k parameter for k-anonymity. Each equivalence class will have
        at least k records.
    generalisation : str, default='midpoint'
        How to generalise values within a partition:
        - 'midpoint': Replace with midpoint of the range
        - 'mean': Replace with actual mean of values in partition
        - 'range': Return (low, high) tuple for each dimension
    relax : bool, default=False
        Use relaxed Mondrian (allows overlap) vs strict Mondrian.
    
    Attributes
    ----------
    partitions_ : List[MondrianPartition]
        The equivalence classes after fitting
    ncp_ : float
        Normalised Certainty Penalty (information loss metric), as percentage
    runtime_ : float
        Time taken to run the algorithm in seconds
    """
    
    def __init__(self, k: int = 5, generalisation: str = 'midpoint', relax: bool = False):
        self.k = k
        self.generalisation = generalisation
        self.relax = relax
        
        # Fitted attributes
        self.partitions_: List[MondrianPartition] = []
        self.ncp_: float = 0.0
        self.runtime_: float = 0.0
        self._qi_range: np.ndarray = None
        self._X: np.ndarray = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Mondrian k-anonymity to data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to anonymise. All features are treated as quasi-identifiers.
        
        Returns
        -------
        X_anon : np.ndarray of shape (n_samples, n_features)
            Anonymised data where records in the same equivalence class
            have identical values.
        """
        self._X = np.asarray(X, dtype=float)
        n_samples, n_features = self._X.shape
        
        if n_samples < self.k:
            raise ValueError(f"Dataset has {n_samples} records but k={self.k}")
        
        # Compute ranges for normalisation
        self._qi_range = self._X.max(axis=0) - self._X.min(axis=0)
        self._qi_range[self._qi_range == 0] = 1  # Avoid division by zero
        
        # Initialise with all records in one partition
        low = self._X.min(axis=0).tolist()
        high = self._X.max(axis=0).tolist()
        initial_partition = MondrianPartition(
            member_indices=list(range(n_samples)),
            low=low,
            high=high,
            n_dims=n_features
        )
        
        # Run Mondrian
        self.partitions_ = []
        start_time = time.time()
        
        if self.relax:
            self._anonymise_relaxed(initial_partition)
        else:
            self._anonymise_strict(initial_partition)
        
        self.runtime_ = time.time() - start_time
        
        # Compute NCP and generate anonymised data
        X_anon = np.zeros_like(self._X)
        total_ncp = 0.0
        
        for partition in self.partitions_:
            # Compute partition NCP
            partition_ncp = 0.0
            for dim in range(n_features):
                width = partition.high[dim] - partition.low[dim]
                partition_ncp += width / self._qi_range[dim]
            partition_ncp *= len(partition)
            total_ncp += partition_ncp
            
            # Generalise records in partition
            indices = partition.member
            if self.generalisation == 'midpoint':
                gen_values = [(partition.low[d] + partition.high[d]) / 2 
                              for d in range(n_features)]
            elif self.generalisation == 'mean':
                gen_values = self._X[indices].mean(axis=0).tolist()
            else:
                # Default to midpoint
                gen_values = [(partition.low[d] + partition.high[d]) / 2 
                              for d in range(n_features)]
            
            for idx in indices:
                X_anon[idx] = gen_values
        
        # Normalise NCP to percentage
        self.ncp_ = (total_ncp / n_features / n_samples) * 100
        
        return X_anon
    
    def _get_normalised_width(self, partition: MondrianPartition, dim: int) -> float:
        """Get normalised width of partition on dimension dim."""
        width = partition.high[dim] - partition.low[dim]
        return width / self._qi_range[dim]
    
    def _choose_dimension(self, partition: MondrianPartition) -> int:
        """Choose dimension with largest normalised width that is allowed to split."""
        n_features = len(partition.low)
        max_width = -1
        max_dim = -1
        
        for dim in range(n_features):
            if not partition.allow[dim]:
                continue
            width = self._get_normalised_width(partition, dim)
            if width > max_width:
                max_width = width
                max_dim = dim
        
        return max_dim
    
    def _find_median(self, partition: MondrianPartition, dim: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Find the median split point for a partition on a dimension.
        
        Returns
        -------
        split_val : float or None
            Value to split on (records <= split_val go left)
        next_val : float or None
            First value in right partition (for updating bounds)
        """
        indices = partition.member
        values = self._X[indices, dim]
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        
        n = len(indices)
        if n < 2 * self.k:
            return None, None
        
        # Find median position ensuring both sides have at least k records
        median_pos = n // 2
        
        # Ensure left side has at least k records
        if median_pos < self.k:
            median_pos = self.k
        
        # Ensure right side has at least k records
        if n - median_pos < self.k:
            median_pos = n - self.k
        
        if median_pos <= 0 or median_pos >= n:
            return None, None
        
        split_val = sorted_values[median_pos - 1]
        next_val = sorted_values[median_pos]
        
        # If split_val == next_val, we can't split here
        if split_val == next_val:
            return None, None
        
        return split_val, next_val
    
    def _anonymise_strict(self, partition: MondrianPartition) -> None:
        """Strict Mondrian: recursively partition without overlap."""
        n_features = len(partition.low)
        
        # Try each dimension
        for _ in range(sum(partition.allow)):
            dim = self._choose_dimension(partition)
            if dim == -1:
                break
            
            split_val, next_val = self._find_median(partition, dim)
            
            if split_val is None:
                partition.allow[dim] = False
                continue
            
            # Split the partition
            indices = partition.member
            values = self._X[indices, dim]
            
            left_mask = values <= split_val
            right_mask = ~left_mask
            
            left_indices = [indices[i] for i in range(len(indices)) if left_mask[i]]
            right_indices = [indices[i] for i in range(len(indices)) if right_mask[i]]
            
            # Check k-anonymity constraint
            if len(left_indices) < self.k or len(right_indices) < self.k:
                partition.allow[dim] = False
                continue
            
            # Create child partitions
            left_high = partition.high.copy()
            left_high[dim] = split_val
            
            right_low = partition.low.copy()
            right_low[dim] = next_val
            
            # Update actual bounds based on data
            left_data = self._X[left_indices]
            right_data = self._X[right_indices]
            
            left_partition = MondrianPartition(
                left_indices,
                left_data.min(axis=0).tolist(),
                left_data.max(axis=0).tolist(),
                n_features
            )
            
            right_partition = MondrianPartition(
                right_indices,
                right_data.min(axis=0).tolist(),
                right_data.max(axis=0).tolist(),
                n_features
            )
            
            # Recurse
            self._anonymise_strict(left_partition)
            self._anonymise_strict(right_partition)
            return
        
        # Cannot split further, add to results
        self.partitions_.append(partition)
    
    def _anonymise_relaxed(self, partition: MondrianPartition) -> None:
        """Relaxed Mondrian: allows overlap to achieve better balance."""
        n_features = len(partition.low)
        
        if not any(partition.allow):
            self.partitions_.append(partition)
            return
        
        dim = self._choose_dimension(partition)
        if dim == -1:
            self.partitions_.append(partition)
            return
        
        split_val, next_val = self._find_median(partition, dim)
        
        if split_val is None:
            partition.allow[dim] = False
            self._anonymise_relaxed(partition)
            return
        
        # Split with handling of median values
        indices = partition.member
        values = self._X[indices, dim]
        
        left_indices = []
        right_indices = []
        mid_indices = []
        
        for i, idx in enumerate(indices):
            val = values[i]
            if val < split_val:
                left_indices.append(idx)
            elif val > split_val:
                right_indices.append(idx)
            else:
                mid_indices.append(idx)
        
        # Distribute mid values to balance partitions
        half_size = len(partition) // 2
        while len(left_indices) < half_size and mid_indices:
            left_indices.append(mid_indices.pop())
        right_indices.extend(mid_indices)
        
        # Check k-anonymity
        if len(left_indices) < self.k or len(right_indices) < self.k:
            partition.allow[dim] = False
            self._anonymise_relaxed(partition)
            return
        
        # Create child partitions with actual data bounds
        left_data = self._X[left_indices]
        right_data = self._X[right_indices]
        
        left_partition = MondrianPartition(
            left_indices,
            left_data.min(axis=0).tolist(),
            left_data.max(axis=0).tolist(),
            n_features
        )
        
        right_partition = MondrianPartition(
            right_indices,
            right_data.min(axis=0).tolist(),
            right_data.max(axis=0).tolist(),
            n_features
        )
        
        self._anonymise_relaxed(left_partition)
        self._anonymise_relaxed(right_partition)
    
    def get_equivalence_classes(self) -> List[np.ndarray]:
        """
        Get the indices of records in each equivalence class.
        
        Returns
        -------
        classes : List[np.ndarray]
            List where each element is an array of record indices
            belonging to that equivalence class.
        """
        return [np.array(p.member) for p in self.partitions_]
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the anonymisation.
        
        Returns
        -------
        stats : dict
            Dictionary containing:
            - k_achieved: Actual minimum equivalence class size (should be >= k)
            - n_eq_classes: Number of equivalence classes
            - ncp: Normalised Certainty Penalty (%)
            - runtime: Time taken (seconds)
            - discernibility: Discernibility metric (sum of squared class sizes)
            - normalised_eq_class_metric: (avg class size) / k_achieved
        """
        if not self.partitions_:
            return {}
        
        sizes = [len(p) for p in self.partitions_]
        k_achieved = min(sizes)
        n_records = sum(sizes)
        n_eq_classes = len(self.partitions_)
        
        discernibility = sum(s ** 2 for s in sizes)
        avg_size = n_records / n_eq_classes
        normalised_eq = avg_size / k_achieved if k_achieved > 0 else float('inf')
        
        return {
            'k_achieved': k_achieved,
            'n_eq_classes': n_eq_classes,
            'ncp': self.ncp_,
            'runtime': self.runtime_,
            'discernibility': discernibility,
            'normalised_eq_class_metric': normalised_eq,
            'mean_class_size': avg_size,
            'median_class_size': float(np.median(sizes)),
        }


def mondrian_anonymise(X: np.ndarray, k: int = 5, generalisation: str = 'midpoint', 
                       relax: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to apply Mondrian k-anonymity.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data to anonymise
    k : int
        k-anonymity parameter
    generalisation : str
        How to generalise ('midpoint' or 'mean')
    relax : bool
        Use relaxed Mondrian
    
    Returns
    -------
    X_anon : np.ndarray
        Anonymised data
    stats : dict
        Statistics about the anonymisation
    """
    anonymiser = MondrianAnonymiser(k=k, generalisation=generalisation, relax=relax)
    X_anon = anonymiser.fit_transform(X)
    stats = anonymiser.get_stats()
    return X_anon, stats
