from qsnow.experiments import SquarePackingExp
from qsnow.interface import Chip, SCTile
from qsnow.helpers import serialize

import argparse
import logging

logger = logging.getLogger(__name__)

def run_and_serialize_spp_experiment(chip: Chip, mean, deviation, distances, data_directory, seed):
    if data_directory:
        serialize.set_data_dir(data_directory)
    if not chip.tag.metadata.get("noise_model", None):
        chip.generate_gaussian_noise(mean, deviation, seed)
    for d in distances:
        logger.info(f'Beginning profiling for distance {d}....')
        new_chip = chip.copy()
        label = f'd{d}_{new_chip.length // 2}x{new_chip.height // 2}_mean_{str(mean).replace('.','_')}'

        exp = SquarePackingExp(new_chip, SCTile(d))
        exp.run()
        exp_path = exp.save(label=label)
        res_path = exp.save_results(label=label)

        logger.info(f"Experiment saved to {exp_path}")
        logger.info(f"Results saved to {res_path}")

def main():
    parser = argparse.ArgumentParser(description="A script that runs Square Packing experiments conveniently.")

    parser.add_argument("--dimensions", nargs=2, type=int, required=True, help="(Length x Height) of the chip in unit cells.")
    parser.add_argument("--mean", type=float, required=True, help="Mean noise to use for gaussian noise model.")
    parser.add_argument("--deviation", type=float, required=True, help="Mean noise to use for gaussian noise model.")
    parser.add_argument("--distances", nargs="+", type=int, required=True, help="Distance of tiles to sample for.")
    parser.add_argument("--directory", type=str, required=False, default=None,  help="Directory to use for saving the experiments and result.")
    parser.add_argument("--seed", type=int, required=False, default=None,  help="Seed for random sampling the gaussian noise. If you use the same seed on two experiments with identical mean and dev., the underlying chip will be identical.")
    parser.add_argument("--logger", type=bool, required=False, default=True,  help="Show additional logging information along progress info.")

    args = parser.parse_args()

    if args.logger:
        logging.basicConfig(level=logging.INFO)

    run_and_serialize_spp_experiment(chip = Chip(args.dimensions[0], args.dimensions[1]), 
                                    mean = args.mean,
                                    deviation = args.deviation,
                                    distances = args.distances,
                                    data_directory = args.directory,
                                    seed = args.seed)

if __name__ == "__main__":
    main()

    