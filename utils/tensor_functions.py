import torch
import numpy as np

def compute_in_batches(f, calc_batch_size, *args, n=None):
    """
    Computes memory heavy function f(*args) in batches
    :param n: the total number of elements, optional if it cannot be determined as args[0].size(0)
    :param f: The function that is computed, should take only tensors as arguments and return tensor or tuple of tensors
    :param calc_batch_size: The batch size to use when computing this function
    :param args: Tensor arguments with equally sized first batch dimension
    :return: f(*args), this should be one or multiple tensors with equally sized first batch dimension
    """
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size  # ceil
    if n_batches == 1:
        return f(*args)

    # Run all batches
    # all_res = [f(*batch_args) for batch_args in zip(*[torch.chunk(arg, n_batches) for arg in args])]
    # We do not use torch.chunk such that it also works for other classes that support slicing
    all_res = [f(*(arg[i * calc_batch_size:(i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    # Allow for functions that return None
    def safe_cat(chunks, dim=0):
        if chunks[0] is None:
            assert all(chunk is None for chunk in chunks)
            return None
        return torch.cat(chunks, dim)

    # Depending on whether the function returned a tuple we need to concatenate each element or only the result
    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(res_chunks, 0) for res_chunks in zip(*all_res))
    return safe_cat(all_res, 0)

def perturb_tensor(tensor, percentage, max_perturbation_degree=0.1):
    """
    Randomly perturbs a percentage of rows in a given tensor to varying degrees.

    Parameters:
    - tensor: torch.Tensor, the tensor to perturb
    - percentage: float, percentage of rows to perturb
    - max_perturbation_degree: float, maximum perturbation degree, default is 0.1
    """
    # Calculate the number of rows to perturb
    num_rows = int(tensor.size(0) * (percentage / 100))
    
    # Select random indices to perturb
    indices_to_perturb = np.random.choice(tensor.size(0), num_rows, replace=False)
    
    for idx in indices_to_perturb:
        # Generate a random perturbation magnitude between 0 and max_perturbation_degree
        perturbation_magnitude = torch.rand(1).item() * max_perturbation_degree
        # Apply the perturbation with the generated magnitude
        perturbation = (torch.rand(tensor[idx].shape) - 0.5) * 2 * perturbation_magnitude
        tensor[idx] = torch.clamp(tensor[idx] + perturbation, 0.0, 1.0)
    
    return tensor
