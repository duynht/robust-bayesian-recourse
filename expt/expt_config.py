import argparse

from expt.config import Config


class Expt5(Config):
    __dictpath__ = "ec.e5"

    all_clfs = ["mlp"]
    all_datasets = ["synthesis"]
    all_methods = ["wachter", "lime_roar", "limels_roar", "rbr"]

    roar_params = {
        "delta_max": 0.2,
    }

    rbr_params = {
        "delta_plus": 0.2,
        "sigma": 1.0,
        "epsilon_op": 0.0,
        "epsilon_pe": 0.0,
    }

    perturb_radius = {
        "synthesis": 0.2,
        "german": 0.2,
        "sba": 0.2,
        "gmc": 0.2,
    }

    params_to_vary = {
        "delta_max": {
            "default": 0.05,
            "min": 0.0,
            "max": 0.2,
            "step": 0.02,
        },
        "epsilon_op": {
            "default": 1.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.5,
        },
        "epsilon_pe": {
            "default": 1.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.5,
        },
        "delta_plus": {
            "default": 0.05,
            "min": 0.0,
            "max": 1.0,
            "step": 0.2,
        },
        "none": {"min": 0.0, "max": 0.0, "step": 0.1},
    }

    kfold = 5
    num_future = 100

    perturb_std = 1.0
    num_samples = 200
    max_ins = 100
    max_distance = 1.0


class ExptConfig(Config):
    __dictpath__ = "ec"

    e5 = Expt5()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument("--dump", default="config.yml", type=str)
    parser.add_argument("--load", default=None, type=str)
    parser.add_argument("--mode", default="merge_cls", type=str)

    args = parser.parse_args()
    if args.load is not None:
        ExptConfig.from_file(args.load)
    ExptConfig.to_file(args.dump, mode=args.mode)
