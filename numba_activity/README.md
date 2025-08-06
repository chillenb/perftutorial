# Your task
Your collaborator is analyzing MD trajectories and complains that your `total_energy` function is slow.

In the system you two are studying, particles are subject to a pairwise repulsive force given by the potential
$$V(r) = \frac{1}{1+r}.$$
Therefore, the total energy is
$$U = \sum_{i < j} \frac{1}{1 + \|\mathbf{r}_i - \mathbf{r}_j \|}.$$

You can estimate the performance of your function (in `kernels.py`) by running `python run.py`.
Try to make this function faster. The performance tips in the [Numba documentation](https://numba.readthedocs.io/en/stable/user/performance-tips.html) will be very useful.
