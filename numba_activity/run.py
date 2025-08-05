from kernels import total_energy, ref_total_energy
import numpy as np

import argparse
import timeit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate total energy of a system.")
    parser.add_argument("-n", "--num_points", type=int, default=2000, help="Number of points in the system")
    parser.add_argument("-r", "--repeats", type=int, default=4, help="Number of repeats for timing")
    args = parser.parse_args()

    coords = np.random.random((3, args.num_points))
    ref_energy = ref_total_energy(coords)
    energy = total_energy(coords)
    if not np.isclose(energy, ref_energy, rtol=1e-6):
        raise ValueError(f"Total energy mismatch: {energy} != {ref_energy}")

    elapsed_time = timeit.timeit("total_energy(coords)", globals=globals(), number=args.repeats)
    print(f"Total energy: {energy}")
    print(f"Average time per run: {elapsed_time / args.repeats} seconds")

    score = (args.num_points ** 2) / (elapsed_time / args.repeats)
    print(f"Performance score: {score:.2e} pairs/second")
