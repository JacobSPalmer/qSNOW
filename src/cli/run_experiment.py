from qsnow.experiments import SquarePackingExp
from qsnow.interface import Chip, SCTile
from qsnow.interface.noise import LogSkewContour, NoiseDistribution, NormalContour, RandomGaussian, SkewContour, Uniform
from qsnow.helpers import serialize
from time import perf_counter

import argparse
import logging

logger = logging.getLogger(__name__)
MODEL_CHOICES = ('gaussian', 'normal-contour', 'skewed-contour', 'log-skewed-contour', 'uniform')
MODEL_ALIASES = {'derived-contour': 'normal-contour'}
DERIVED = 'derived'  # couplers follow their endpoints; the default and the pre-flag behaviour


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def distribution_from_flags(model, location, deviation=0.0, skew=0.0, center=None, seed=None) -> NoiseDistribution:
    match resolve_model(model):
        case 'gaussian':
            return RandomGaussian(location, deviation, seed=seed)
        case 'normal-contour':
            return NormalContour(location, deviation, seed=seed)
        case 'skewed-contour':
            return SkewContour(location, deviation, skew, center or SkewContour.center, seed=seed)
        case 'log-skewed-contour':
            # `location` is a rate (median or geometric mean); `deviation`/`skew` describe log10(rate)
            return LogSkewContour(location, deviation, skew, center or LogSkewContour.center, seed=seed)
        case 'uniform':
            return Uniform(location)
        case _:
            raise ValueError(f'Noise model {model!r} either not supported or unknown; choose from {MODEL_CHOICES}.')


def run_and_serialize_spp_experiment(chip: Chip, *, location, deviation, distances, data_directory, additional_label, seed, shots, max_errors, noise_model, min_errors, shot_ceiling, skew=0.0, center=None,
                                     coupler_model=DERIVED, coupler_location=None, coupler_deviation=0.0, coupler_skew=0.0, coupler_center=None, coupler_seed=None, correlation=None):
    if data_directory:
        serialize.set_data_dir(data_directory)
    logger.info(f'Creating chip with {noise_model} noise model using {seed}')
    if chip.spec.noise_model is None:
        chip.generate_noise(distribution_from_flags(noise_model, location, deviation, skew, center, seed))
    # Couplers: derived from their endpoints unless a model of their own is asked for. A
    # chip loaded from a flake that already carries one is used as saved, like the sites.
    if coupler_model != DERIVED and chip.spec.coupler_model is None:
        if coupler_location is None:
            raise ValueError('coupler_location is required when coupler_model is not derived.')
        logger.info(f'Creating coupler landscape with {coupler_model} noise model using {coupler_seed} (correlation={correlation})')
        chip.generate_coupler_noise(
            distribution_from_flags(coupler_model, coupler_location, coupler_deviation, coupler_skew, coupler_center, coupler_seed),
            correlation=correlation,
        )
    start = perf_counter()
    for d in distances:
        logger.info(f'Beginning profiling for distance {d}....')
        new_chip = chip.copy()
        label = (f'{additional_label}_' if additional_label else '') + f'd{d}_{new_chip.unit_dims[0]}x{new_chip.unit_dims[1]}_loc_{str(location).replace('.','_')}'

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
    # Flags left at None are omitted rather than written as "None": every such flag
    # defaults to None, so a replay resolves them the same way this run did. Seeds and
    # centres are written as the chip resolved them, so the replay is identical.
    resolved = {
        'seed': getattr(chip.spec.noise_model, 'seed', None),
        'center': getattr(chip.spec.noise_model, 'center', None),
        'coupler_seed': getattr(chip.spec.coupler_model, 'seed', None),
        'coupler_center': getattr(chip.spec.coupler_model, 'center', None),
    }
    config_arr = []
    filename = args.pop('name')
    config_arr.append(format_str('name', filename))
    for k, v in args.items():
        v = resolved.get(k, v)
        if v is not None:
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

    def str_or_none(value):
        # a hand-written config may spell an absent optional as the literal "None"
        return None if value == "None" else value

    def unit_float_or_none(value):
        if value == "None":
            return None
        rho = float(value)
        if not 0.0 <= rho <= 1.0:
            raise argparse.ArgumentTypeError(f"correlation must lie in [0, 1], given {value}")
        return rho
    
    parser = argparse.ArgumentParser(description="A script that runs Square Packing experiments conveniently.\n The arguments can either be specified individually or passed from a text file using `@<path/to/text_file>.", 
                                     fromfile_prefix_chars='@')
    
    parser.add_argument("--name", type=str, required=False, default=None, help="Name of the experiment. This is appended to the beginning of the saved flake filenames.")
    parser.add_argument("--dimensions", nargs=2, type=int, required=False, default=(10,10), help="(Length x Height) of the chip in unit cells.")
    parser.add_argument("--model", choices=[*MODEL_CHOICES, *MODEL_ALIASES], default='gaussian', metavar=f"{{{','.join(MODEL_CHOICES)}}}", help="The noise distribution of the chip's qubits.")
    parser.add_argument("--location", "--mean", dest="location", type=float, required=True, help="Location of the PER distribution: the mean (or median, see --center) for gaussian/contour models; for log-skewed-contour the median rate; for uniform the single PER value. --mean is accepted as an alias for saved configs.")
    parser.add_argument("--deviation", type=float, required=False, default=0.0, help="Spread of the PER distribution: a standard deviation in rate units, or for log-skewed-contour the standard deviation of log10(rate) in decades.")
    parser.add_argument("--skew", type=float, required=False, default=0.0, help="Skewness of the PER distribution (skewed-contour), or of log10(rate) (log-skewed-contour). Positive is right-skewed; 0 is normal / log-normal.")
    parser.add_argument("--center", choices=['mean', 'median', 'geometric'], required=False, default=None, help="Which statistic --location pins: mean (default) or median for skewed-contour; median (default) or geometric for log-skewed-contour.")
    parser.add_argument("--distances", nargs="+", type=int, required=True, help="Distance of tiles to sample for.")
    parser.add_argument("--directory", type=str, required=False, default=None,  help="Directory to use for saving the experiments and result.")
    parser.add_argument("--seed", type=int_or_none, required=False, default=None,  help="Seed for the qubit noise model. The same seed with identical model parameters reproduces the identical chip; None draws a fresh seed, which --save_config records.")
    parser.add_argument("--coupler_model", choices=[DERIVED, *MODEL_CHOICES, *MODEL_ALIASES], default=DERIVED, metavar=f"{{{','.join((DERIVED, *MODEL_CHOICES))}}}", help="The noise distribution of the chip's couplers. 'derived' (default) keeps each coupler at a function of its two qubits; any other choice gives the couplers a landscape of their own.")
    parser.add_argument("--coupler_location", "--coupler_mean", dest="coupler_location", type=float, required=False, default=None, help="Location of the coupler PER distribution, as --location is for the qubits. Required unless --coupler_model is derived.")
    parser.add_argument("--coupler_deviation", type=float, required=False, default=0.0, help="Spread of the coupler PER distribution, as --deviation is for the qubits.")
    parser.add_argument("--coupler_skew", type=float, required=False, default=0.0, help="Skewness of the coupler PER distribution, as --skew is for the qubits.")
    parser.add_argument("--coupler_center", choices=['mean', 'median', 'geometric'], required=False, default=None, help="Which statistic --coupler_location pins, as --center is for the qubits.")
    parser.add_argument("--coupler_seed", type=int_or_none, required=False, default=None, help="Seed for the coupler noise model; None draws a fresh one, which --save_config records.")
    parser.add_argument("--correlation", type=unit_float_or_none, required=False, default=None, help="Correlation in [0, 1] between a coupler and the mean of its two qubits, in latent terms: 1 reproduces the endpoint-mean ordering, 0 is independent of the qubits, None applies the coupler model uncorrelated.")
    parser.add_argument("--shots", type=int, required=False, default=50_000, help="Number of shots to sample for each candidate position on a chip.")
    parser.add_argument("--max_errors", type=int_or_none, required=False, default=None, help = "Maximum number of errors encountered before exiting sampling. If the number of errors sampled surpasses this value then the sampling will stop regardless of maximum shot count.")
    parser.add_argument("--min_errors", type=int, required=False, default=30, help="Minimum number of errors that should be encountered at each location. If a sample reaches the provided shot count without sampling at least this many errors, it will continue until a maximum shot ceiling. Default value is 30.")
    parser.add_argument("--shot_ceiling", type=int_or_none, required = False, default=None, help="If minimum errors is not None, then providing shot ceiling here determines the upper bounds of shots in order to sample the minimum errors. This defaults to 20x the provided ideal shot count.")
    parser.add_argument("--logger", type=bool, required=False, default=True,  help="Show additional logging information along progress info.")
    parser.add_argument("--save_config", type=bool, required=False, default=False, help="Exports the command arguements to a reusable <name>_config.txt file that can be used to identically run the experiment.")
    parser.add_argument("--chip", type=str_or_none, required=False, default=None, help="Path to a flake file containing a chip. If specified, the provided dimensions and noise parameters will be ignored")

    args = parser.parse_args()

    if args.save_config and args.name is None:
        parser.error("--name is required when --save_config is set.")
    if args.coupler_model != DERIVED and args.coupler_location is None:
        parser.error("--coupler_location is required when --coupler_model is not derived.")

    if args.logger:
        logging.basicConfig(level=logging.INFO)

    if args.chip:
        chip = serialize.import_flake(args.chip)
        if not chip:
             parser.error(f"path at {args.chip} provided for --chip yielded no valid chip flake file.") 
    else:
        chip = Chip(args.dimensions[0], args.dimensions[1])

    chip = run_and_serialize_spp_experiment(chip = chip, 
                                    location = args.location,
                                    deviation = args.deviation,
                                    distances = args.distances,
                                    data_directory = args.directory,
                                    additional_label = args.name,
                                    shots = args.shots,
                                    seed = args.seed,
                                    noise_model = args.model,
                                    skew = args.skew,
                                    center = args.center,
                                    coupler_model = args.coupler_model,
                                    coupler_location = args.coupler_location,
                                    coupler_deviation = args.coupler_deviation,
                                    coupler_skew = args.coupler_skew,
                                    coupler_center = args.coupler_center,
                                    coupler_seed = args.coupler_seed,
                                    correlation = args.correlation,
                                    max_errors = args.max_errors,
                                    min_errors = args.min_errors,
                                    shot_ceiling = args.shot_ceiling)
    if args.save_config:
        save_configuration(vars(args), chip, args.name)

if __name__ == "__main__":
    main()

    