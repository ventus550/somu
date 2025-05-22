# The code here is an unoptimized torch version of the true rust
# implementation included for the simplest experiment reproducibility


from typing import Sequence

import numpy
import torch
from tqdm import tqdm

Matrix = Sequence[Sequence[float]]


def generate_grid(dims):
    grid = torch.meshgrid([torch.arange(d) for d in dims])
    return torch.stack([g.flatten() for g in grid], dim=-1)


def compute_distance_matrix(p, q):
    p = p.unsqueeze(1)  # Shape (n_p, 1, d_p)
    q = q.unsqueeze(0)  # Shape (1, n_q, d_q)
    dist_matrix = torch.norm(p - q, dim=2)
    return dist_matrix


def sample_rows(arr, n):
    indices = torch.randperm(arr.shape[0])[:n]
    return arr[indices]


def organize(x, influence, dst):
    # dst: centroids x data points
    bmu = torch.argmin(dst, dim=1)  # Best Matching Units

    # Vectorized: Compute the influence rows for each data point
    influence_rows = influence[bmu]  # Shape (n, m)

    # Now compute the numerator and denominator in a vectorized manner
    # Broadcast influence_rows with x (data points)
    # Influence is a weight for each data point and each neuron
    numerator = torch.matmul(influence_rows.T, x)  # Shape (m, d)

    # Sum the influence values to get the denominator
    denominator = influence_rows.sum(dim=0)  # Shape (m,)

    # Normalize the numerator by the denominator
    return numerator / (denominator[:, None] + 1e-8)


def som(
    data: Matrix,
    units: Sequence | Matrix,
    iters=1,
    sigma_initial=1.0,
    batch_size=1,
    seed=None,
) -> numpy.ndarray:
    """
    Train a Self-Organizing Map (SOM) on the given data.

    Args:
        data (np.ndarray): The input data for training the SOM.
        units (int): The matrix-defined topology of the SOM or dimensions of a regular grid topology.
        iters (int, optional): The number of training iterations (batches). Default is 1.
        sigma_initial (float, optional): The initial neighborhood radius. Default is 1.0.
        batch_size (int, optional): The number of samples to process in each batch. Default is 1.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        np.ndarray: The trained SOM weights (centroids)
    """

    data = torch.tensor(data)
    units = torch.tensor(units)

    if seed:
        torch.manual_seed(seed)

    if len(units.shape) == 1:
        units = generate_grid(units).double()

    m = units.shape[0]
    centroids = sample_rows(data, m)
    unit_dst = compute_distance_matrix(units, units)

    for epoch in tqdm(range(iters)):
        sigma = sigma_initial * torch.exp(
            -torch.tensor(epoch, dtype=torch.float32) / iters
        )
        influence = torch.exp(-(unit_dst**2) / (2 * sigma**2))
        batch = sample_rows(data, batch_size)
        dst = compute_distance_matrix(batch, centroids)
        centroids = organize(batch, influence, dst)

    return centroids.cpu().detach().numpy()
