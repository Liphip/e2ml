"""
Latin hypercube with normall distribution for farther values on the sides
"""

import numpy as np

from sklearn.utils import check_scalar, check_array, check_consistent_length, column_or_1d

def lat_hyp_cube_norm_dist_unit(n_samples, n_dimensions, mus, sigmas, features, _random_state=42):
   """
   Generate a combination of a latin-hypercube with normall distribution

   Parameters
   ----------
   n_samples : int
      Number of samples to be generated.

   n_dimensions : int
      Dimensionality of the generated samples.

   mus : array-like of shape (n_dimensions, 1)
      `mus[d]` is the mean value for dimension `d`.

   sigmas : array-like of shape (n_dimensions, 1)
      `sigmas[d]` is the standard deviation for dimension `d`.   
      
   Returns
   -------
   X : np.ndarray of shape (n_samples, n_dimensions)
       An `n_samples-by-n_dimensions` design matrix whose levels are spaced between zero and one.
   """
   check_scalar(n_samples, name="n_samples", target_type=int, min_val=1)
   check_scalar(n_dimensions, name="n_dimensions", target_type=int, min_val=1)
   check_consistent_length(mus, sigmas)
   check_consistent_length(mus, features)
   rand = np.random.RandomState(_random_state)
   X = np.zeros((n_samples, n_dimensions))
   for idx, feature in enumerate(features):
         X[:, idx] = rand.normal(loc=mus[feature], scale=sigmas[feature], size=n_samples)
   return X
   


def lat_hyp_cube_norm_dist(n_samples, n_dimensions, mus, sigmas, features, bounds=None, min_max=False, min_max_samples=100):
   """
   Generate a specified number of samples according to a Latin hypercube in a user-specified bounds.

   Parameters
   ----------
   n_samples : int
      Number of samples to be generated.

   n_dimensions : int
      Dimensionality of the generated samples.

   mus : array-like of shape (n_dimensions, 1)
      `mus[d]` is the mean value for dimension `d`.

   sigmas : array-like of shape (n_dimensions, 1)
      `sigmas[d]` is the standard deviation for dimension `d`.
      
   bounds : None or array-like of shape (n_dimensions, 2)
      `bounds[d, 0]` is the minimum and `bounds[d, 1]` the maximum
      value for dimension `d`.

   min_max : bool
      If True, the minimum and maximum values of the normal 
      distribution must be included in the samples.

   min_max_samples : int
      Number of samples to be generated to find the minimum and 
      maximum values of the normal distribution.

   Returns
   -------
   X : numpy.ndarray of shape (n_samples, n_dimensions)
      Generated samples.
   """
   def check_sample(sample, bounds, mus, sigmas, _random_state=42):
         """
         Check if a sample is valid.

         Parameters
         ----------
         sample : array-like of shape (n_dimensions, 1)
            Sample (roW) to be checked.

         bounds : array-like of shape (n_dimensions, 2)
            `bounds[d, 0]` is the minimum and `bounds[d, 1]` the maximum
            value for dimension `d`.

         mus : array-like of shape (n_dimensions, 1)
            `mus[d]` is the mean value for dimension `d`.
            
         sigmas : array-like of shape (n_dimensions, 1)
            `sigmas[d]` is the standard deviation for dimension `d`.

         Returns
         -------  
         sample : array-like of shape (n_dimensions, 1)
            Valid sample.
         """
         for idx, feature in enumerate(features):
               if bounds[idx,0] is not None:
                  if sample[idx] <= bounds[idx, 0]:
                     sample[idx] = np.random.choice(np.random.normal(loc=mus[feature], scale=sigmas[feature], size=n_samples * 2), 1)
                     return check_sample(sample, bounds, mus, sigmas)
               elif bounds[idx,0] is not None:
                  if sample[idx] >= bounds[idx, 1]:
                        sample[idx] = np.random.choice(np.random.normal(loc=mus[feature], scale=sigmas[feature], size=n_samples * 2), 1)
                        return check_sample(sample, bounds, mus, sigmas)
               print(sample) if sample[idx] <= 0 else None
         return sample

   # Check parameters.
   check_scalar(n_samples, name="n_samples", target_type=int, min_val=1)
   check_scalar(n_dimensions, name="n_dimensions", target_type=int, min_val=1)
   check_consistent_length(mus, sigmas)
   check_consistent_length(mus, bounds)
   check_consistent_length(mus, features)
   if bounds is not None:
       if bounds.shape[0] != n_dimensions or bounds.shape[1] != 2:
           raise ValueError("`bounds` must have shape `(n_dimensions, 2)`.")
   else:
       bounds = np.zeros((n_dimensions, 2))
       bounds[:, 1] = 1

   # Generate samples.
   if min_max:
      if bounds is None:
            raise ValueError("If `min_max` is True, `bounds` must be specified.")
      get_min = all(v is not None for v in bounds[:, 0])
      get_max = all(v is not None for v in bounds[:, 1])
      X = (lat_hyp_cube_norm_dist_unit(n_samples-2, n_dimensions, mus, sigmas, features) if get_min and get_max else lat_hyp_cube_norm_dist_unit(n_samples, n_dimensions, mus, sigmas, features)) if get_min or get_max else lat_hyp_cube_norm_dist_unit(n_samples, n_dimensions, mus, sigmas, features)
      min_sample = np.zeros((1, n_dimensions))
      max_sample = np.zeros((1, n_dimensions))
      for idx, feature in enumerate(features):
            dist = np.random.normal(loc=mus[feature], scale=sigmas[feature], size=min_max_samples)
            min_sample[0, idx] = (np.min(dist) if np.min(dist) < bounds[idx, 0] else bounds[idx, 0] + 1/sigmas[feature]) if bounds[idx, 0] is not None else np.random.normal(loc=mus[feature], scale=sigmas[feature], size=1)
            max_sample[0, idx] = (np.max(dist) if np.max(dist) > bounds[idx, 1] else bounds[idx, 1] - 1/sigmas[feature]) if bounds[idx, 1] is not None else np.random.normal(loc=mus[feature], scale=sigmas[feature], size=1)
      X = (np.concatenate((min_sample, X, max_sample), axis=0) if get_min and get_max else np.concatenate((X, max_sample), axis=0)) if get_max else np.concatenate((min_sample, X), axis=0) if get_min else X
   else:
      X = lat_hyp_cube_norm_dist_unit(n_samples, n_dimensions, mus, sigmas, features)
   for i in range(n_samples):
         X[i, :] = check_sample(X[i, :], bounds, mus, sigmas)
   return X 
