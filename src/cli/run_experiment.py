from qsnow.experiments import SquarePackingExp
from qsnow.interface import Chip, SCTile
from qsnow.helpers import serialize
from time import perf_counter

import argparse
import logging

logger = logging.getLogger(__name__)

def run_and_serialize_spp_experiment(chip: Chip, *, mean, deviation, distances, data_directory, additional_label, seed, shots, max_errors, noise_model, min_errors, shot_ceiling):
    if data_directory:
        serialize.set_data_dir(data_directory)
    logger.info(f'Creating chip with {noise_model} noise model using {seed}')
    if not chip.tag.metadata.get("noise_model", None):
        match noise_model:
            case 'gaussian':
                chip.generate_gaussian_noise(mean, deviation, seed)
            case 'derived-contour':
                chip.generate_derived_contour_noise(mean, deviation, seed)
            case 'uniform':
                chip.generate_uniform_noise(mean)
            case _:
                raise ValueError(f'Noise model {noise_model} either not supported or unknown.')
    start = perf_counter()
    for d in distances:
        logger.info(f'Beginning profiling for distance {d}....')
        new_chip = chip.copy()
        label = (f'{additional_label}_' if additional_label else '') + f'd{d}_{new_chip.unit_dims[0]}x{new_chip.unit_dims[1]}_mean_{str(mean).replace('.','_')}'

        exp = SquarePackingExp(new_chip, SCTile(d))
        exp.tag.name = additional_label + f'_d{d}'
        exp.run(shots=shots, max_errors=max_errors, min_errors=min_errors, max_topup_shots=shot_ceiling)
        exp_path = exp.save(label=label)
        res_path = exp.save_results(label=label)

        logger.info(f"Experiment saved to {exp_path}")
        logger.info(f"Results saved to {res_path}")
    m, s  = divmod(perf_counter() - start, 60)
    time_str = f'{int(m)} mins, {s:.2g} secs' if m else f'{s:.2g} secs'
    logger.info(f'Completed profiling for {len(distances)} distance(s) in {time_str}')
    return chip

def save_configuration(args: dict, chip: Chip, name: str):
    def format_str(name, values):
        if isinstance(values, list):
            return f'--{name}\n{"\n".join([str(v) for v in values])}\n'
        else:
            return f'--{name}\n{values}\n'
    config_arr = []
    filename = args.pop('name')
    config_arr.append(format_str('name', filename))
    for k, v in args.items():
        match k:
            case 'seed':
                config_arr.append(format_str(k, chip.tag.metadata.get('noise_model', {}).get('seed', None)))
            case _:
                config_arr.append(format_str(k, v))

    filepath = serialize.get_data_dir().joinpath(f'{filename}.txt')
    with open(filepath, 'w') as file:
        file.writelines(config_arr)

    logger.info(f'Run configuration saved to {filepath}')

    
def main():
    def int_or_none(value):
        if value == "None":
            return None
        return int(value)
    
    parser = argparse.ArgumentParser(description="A script that runs Square Packing experiments conveniently.\n The arguments can either be specified individually or passed from a text file using `@<path/to/text_file>.", 
                                     fromfile_prefix_chars='@')
    
    parser.add_argument("--name", type=str, required=False, default=None, help="Name of the experiment. This is appended to the beginning of the saved flake filenames.")
    parser.add_argument("--dimensions", nargs=2, type=int, required=True, help="(Length x Height) of the chip in unit cells.")
    parser.add_argument("--model", choices=['gaussian', 'derived-contour', 'uniform'], default="The noise distribution of the chip.")
    parser.add_argument("--mean", type=float, required=True, help="Mean noise to use for noise model. If uniform distribution is selected, then the mean ")
    parser.add_argument("--deviation", type=float, required=False, default=0.0, help="Deviation of PER noise distribution to use for noise model.")
    parser.add_argument("--distances", nargs="+", type=int, required=True, help="Distance of tiles to sample for.")
    parser.add_argument("--directory", type=str, required=False, default=None,  help="Directory to use for saving the experiments and result.")
    parser.add_argument("--seed", type=int_or_none, required=False, default=None,  help="Seed for random sampling the gaussian noise. If you use the same seed on two experiments with identical mean and dev., the underlying chip will be identical.")
    parser.add_argument("--shots", type=int, required=False, default=50_000, help="Number of shots to sample for each candidate position on a chip.")
    parser.add_argument("--max_errors", type=int_or_none, required=False, default=None, help = "Maximum number of errors encountered before exiting sampling. If the number of errors sampled surpasses this value then the sampling will stop regardless of maximum shot count.")
    parser.add_argument("--min_errors", type=int, required=False, default=30, help="Minimum number of errors that should be encountered at each location. If a sample reaches the provided shot count without sampling at least this many errors, it will continue until a maximum shot ceiling. Default value is 30.")
    parser.add_argument("--shot_ceiling", type=int_or_none, required = False, default=None, help="If minimum errors is not None, then providing shot ceiling here determines the upper bounds of shots in order to sample the minimum errors. This defaults to 20x the provided ideal shot count.")
    parser.add_argument("--logger", type=bool, required=False, default=True,  help="Show additional logging information along progress info.")
    parser.add_argument("--save_config", type=bool, required=False, default=False, help="Exports the command arguements to a reusable <name>_config.txt file that can be used to identically run the experiment.")

    args = parser.parse_args()

    if args.save_config and args.name is None:
        parser.error("--name is required when --save_config is set.")

    if args.logger:
        logging.basicConfig(level=logging.INFO)

    chip = run_and_serialize_spp_experiment(chip = Chip(args.dimensions[0], args.dimensions[1]), 
                                    mean = args.mean,
                                    deviation = args.deviation,
                                    distances = args.distances,
                                    data_directory = args.directory,
                                    additional_label = args.name,
                                    shots = args.shots,
                                    seed = args.seed,
                                    noise_model = args.model,
                                    max_errors = args.max_errors,
                                    min_errors = args.min_errors,
                                    shot_ceiling = args.shot_ceiling)
    if args.save_config:
        save_configuration(vars(args), chip, args.name)

if __name__ == "__main__":
    main()

    