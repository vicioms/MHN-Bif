import torch
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Union

def _find_prototypes_random_projection_numpy(x : NDArray, num_random_projections : int, max_num_prototypes : Optional[int] = None, seed_or_rng : Union[None, np.random.Generator, int] = None) -> Tuple[NDArray, NDArray]:
    """
        Find prototypes in a dataset using random projections. 
        The procedure is described in "https://arxiv.org/pdf/1405.4275". 

        Parameters
        ----------
        x : NDArray
            A 2D array of shape (n_samples, n_features) representing the dataset.
        num_random_projections : int
            The number of random projections to use for finding prototypes.
        max_num_prototypes : Optional[int], default=None
            The maximum number of prototypes to return. If None, all prototypes found will be returned.
        seed_or_rng : Union[None, np.random.Generator, int], default=None
            A random number generator or seed for reproducibility. If None, a new generator will be created.

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing:
            - An array of indices of the selected prototypes in the original dataset.
            - An array of counts indicating how many times each prototype was selected across the random projections.
    """

    if len(x.shape) != 2:
        raise ValueError(f"Input array must be 2D, but got shape {x.shape}")
    n, p = x.shape
    if seed_or_rng is None:
        rng = np.random.default_rng()
    elif isinstance(seed_or_rng, int):
        rng = np.random.default_rng(seed=seed_or_rng)

    g = rng.standard_normal(size=(p, num_random_projections))

    y = x @ g

    i_plus = np.argmax(y, axis=0)
    i_minus = np.argmin(y, axis=0)

    indices = np.concatenate((i_plus, i_minus))

    prototype_indices, counts = np.unique(indices,return_counts=True)

    order = np.argsort(counts)[::-1]

    if max_num_prototypes is not None and max_num_prototypes < len(prototype_indices):
        order = order[:max_num_prototypes]

    return prototype_indices[order], counts[order]
def _find_prototypes_random_projection_torch(x : torch.Tensor, num_random_projections : int, max_num_prototypes : Optional[int] = None, seed_or_rng : Optional[Union[torch.Generator, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Find prototypes in a dataset using random projections. 
        The procedure is described in "https://arxiv.org/pdf/1405.4275". 

        Parameters
        ----------
        x : torch.Tensor
            A 2D array of shape (n_samples, n_features) representing the dataset.
        num_random_projections : int
            The number of random projections to use for finding prototypes.
        max_num_prototypes : Optional[int], default=None
            The maximum number of prototypes to return. If None, all prototypes found will be returned.
        seed_or_rng : Optional[Union[torch.Generator, int]], default=None
            A random seed or generator for reproducibility.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - An array of indices of the selected prototypes in the original dataset.
            - An array of counts indicating how many times each prototype was selected across the random projections.
    """

    if len(x.shape) != 2:
        raise ValueError(f"Input array must be 2D, but got shape {x.shape}")
    n, p = x.shape
    if seed_or_rng is not None:
        if isinstance(seed_or_rng, int):
            torch.manual_seed(seed_or_rng)
        else:
            torch.set_rng_state(seed_or_rng.get_state())

    g = torch.randn(p, num_random_projections, device=x.device)

    y = x @ g

    i_plus = torch.argmax(y, dim=0)
    i_minus = torch.argmin(y, dim=0)

    indices = torch.stack((i_plus, i_minus))

    prototype_indices, counts = torch.unique(indices, return_counts=True)

    order = torch.argsort(counts, descending=True)

    if max_num_prototypes is not None and max_num_prototypes < len(prototype_indices):
        order = order[:max_num_prototypes]

    return prototype_indices[order], counts[order]
def find_prototypes_random_projection(x : Union[NDArray, torch.Tensor], num_random_projections : int, max_num_prototypes : Optional[int] = None, rng_or_seed : Optional[Union[np.random.Generator, torch.Generator, int]] = None) -> Tuple[Union[NDArray, torch.Tensor], Union[NDArray, torch.Tensor]]:
    """
        Find prototypes in a dataset using random projections. 
        The procedure is described in "https://arxiv.org/pdf/1405.4275". 

        Parameters
        ----------
        x : Union[NDArray, torch.Tensor]
            A 2D array of shape (n_samples, n_features) representing the dataset.
        num_random_projections : int
            The number of random projections to use for finding prototypes.
        max_num_prototypes : Optional[int], default=None
            The maximum number of prototypes to return. If None, all prototypes found will be returned.
        rng_or_seed : Optional[Union[np.random.Generator, torch.Generator, int]], default=None
            A random number generator or seed for reproducibility. If None, a new generator will be created.


        Returns
        -------
        Tuple[Union[NDArray, torch.Tensor], Union[NDArray, torch.Tensor]]
            A tuple containing:
            - An array of indices of the selected prototypes in the original dataset.
            - An array of counts indicating how many times each prototype was selected across the random projections.
    """
    if isinstance(x, np.ndarray):
        if rng_or_seed is not None and isinstance(rng_or_seed, torch.Generator):
            raise TypeError("Expected a numpy random generator or seed for numpy input, but got a torch Generator.")
        return _find_prototypes_random_projection_numpy(x, num_random_projections, max_num_prototypes, rng_or_seed)
    elif isinstance(x, torch.Tensor):
        if rng_or_seed is not None and isinstance(rng_or_seed, np.random.Generator):
            raise TypeError("Expected a torch random generator or seed for torch input, but got a numpy Generator.")
        return _find_prototypes_random_projection_torch(x, num_random_projections, max_num_prototypes, rng_or_seed)
    else:
        raise TypeError(f"Input array must be either a numpy ndarray or a torch Tensor, but got {type(x)}")


