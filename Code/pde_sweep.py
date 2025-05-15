import os
import json
from tqdm import tqdm
from copy import deepcopy

def sweep(pde_func, path_to_params, sweep_keys, sweep_factors, output_directory):
    print("Initializing...")

    # Load parameters from JSON file
    with open(path_to_params, 'r') as f:
        base_params = json.load(f)

    # make directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # write parameters to directory
    with open(f"{output_directory}/params.json", 'w') as outfile:
            json.dump(base_params, outfile, indent=4)

    print("Running control...")
    # Run control
    pde_func(base_params, f"{output_directory}/control.gif")

    with tqdm(total=len(sweep_keys) * len(sweep_factors), desc="Sweeping parameters") as pbar:
        for key in sweep_keys:
            for factor in sweep_factors:
                pbar.set_postfix_str(f"{key} x{factor:.1f}")
                params = deepcopy(base_params)
                params[key] *= factor
                pde_func(params, f"{output_directory}/{key}_x{factor:.1f}.gif")
                pbar.update(1)

    print("Sweep complete")