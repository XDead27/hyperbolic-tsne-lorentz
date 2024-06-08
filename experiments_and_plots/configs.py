"""
This file contains different configurations for experiments to be run on.
"""

from dotenv import load_dotenv
import os

from hyperbolicTSNE import SequentialOptimizer


class TSNEConfigs:
    
    def get_all_configs(self):
        return [
            self.config_accelerated_poincare,
            self.config_exact_poincare,
            self.config_accelerated_lorentz,
            self.config_exact_lorentz,
            self.config_accelerated_lorentz_centroid,
        ]

    """
    Poincare Quadtree Configuration
    """
    def config_accelerated_poincare_get_opt_params(self, n_samples):
        config_choice = self.config_accelerated_poincare
        val = n_samples / (config_choice["opt_params"]["exaggeration"] * 1000)

        config_choice["opt_params"]["learning_rate_ex"] = val
        config_choice["opt_params"]["learning_rate_main"] = val
        return config_choice["opt_type"](**config_choice["opt_params"])

    config_accelerated_poincare = {
        "config_id": 1000,
        "name": "Quadtree Poincare",
        "data_num_components": 2,
        "opt_type": SequentialOptimizer.sequence_poincare,
        "opt_params": {
            "learning_rate_ex": None,
            "learning_rate_main": None,
            "exaggeration": 12,
            "vanilla": False,
            "momentum_ex": 0.5,
            "momentum": 0.8,
            "exact": False,
            "area_split": False,
            "n_iter_check": 10,
            "hyperbolic_model": "poincare",
            "size_tol": 0.999,
        },
        "get_opt_params": config_accelerated_poincare_get_opt_params,
    }

    """
    Poincare Exact Configuration
    """
    def config_exact_poincare_get_opt_params(self, n_samples):
        config_choice = self.config_exact_poincare
        val = n_samples / (config_choice["opt_params"]["exaggeration"] * 1000)

        config_choice["opt_params"]["learning_rate_ex"] = val
        config_choice["opt_params"]["learning_rate_main"] = val
        return config_choice["opt_type"](**config_choice["opt_params"])

    config_exact_poincare = {
        "config_id": 1010,
        "name": "Exact Poincare",
        "data_num_components": 2,
        "opt_type": SequentialOptimizer.sequence_poincare,
        "opt_params": {
            "learning_rate_ex": None,
            "learning_rate_main": None,
            "exaggeration": 12,
            "vanilla": False,
            "momentum_ex": 0.5,
            "momentum": 0.8,
            "exact": True,
            "area_split": False,
            "n_iter_check": 10,
            "hyperbolic_model": "poincare",
            "size_tol": 0.999,
        },
        "get_opt_params": config_exact_poincare_get_opt_params,
    }

    """
    Lorentz Hyperboloid Configuration
    """
    def config_accelerated_lorentz_get_opt_params(self, n_samples):
        config_choice = self.config_accelerated_lorentz
        val = n_samples / (config_choice["opt_params"]["exaggeration"] * 5)

        config_choice["opt_params"]["learning_rate_ex"] = val / 12
        config_choice["opt_params"]["learning_rate_main"] = val
        return config_choice["opt_type"](**config_choice["opt_params"])

    config_accelerated_lorentz = {
        "config_id": 1100,
        "name": "Octree Lorentz",
        "data_num_components": 3,
        "opt_type": SequentialOptimizer.sequence_lorentz_proj,
        "opt_params": {
            "learning_rate_ex": None,
            "learning_rate_main": None,
            "exaggeration": 12,
            "vanilla": False,
            "momentum_ex": 0.35,
            "momentum": 0.6,
            "exact": False,
            "area_split": False,
            "n_iter_check": 10,
            "hyperbolic_model": "lorentz",
            "size_tol": 0.96,
        },
        "get_opt_params": config_accelerated_lorentz_get_opt_params,
    }

    """
    Lorentz Hyperboloid Exact
    """
    def config_exact_lorentz_get_opt_params(self, n_samples):
        config_choice = self.config_exact_lorentz
        val = n_samples / (config_choice["opt_params"]["exaggeration"] * 1000)

        config_choice["opt_params"]["learning_rate_ex"] = val
        config_choice["opt_params"]["learning_rate_main"] = val
        return config_choice["opt_type"](**config_choice["opt_params"])

    config_exact_lorentz = {
        "config_id": 1110,
        "name": "Exact Lorentz",
        "data_num_components": 3,
        "opt_type": SequentialOptimizer.sequence_lorentz_proj,
        "opt_params": {
            "learning_rate_ex": None,
            "learning_rate_main": None,
            "exaggeration": 12,
            "vanilla": False,
            "momentum_ex": 0.35,
            "momentum": 0.6,
            "exact": True,
            "area_split": False,
            "n_iter_check": 10,
            "hyperbolic_model": "lorentz",
            "size_tol": 0.96,
        },
        "get_opt_params": config_exact_lorentz_get_opt_params,
    }

    """
    Accelerated Lorentz Centroid Configuration
    """
    def config_accelerated_lorentz_centroid_get_opt_params(self, n_samples):
        config_choice = self.config_accelerated_lorentz_centroid
        val = n_samples / (config_choice["opt_params"]["exaggeration"] * 5)

        config_choice["opt_params"]["learning_rate_ex"] = val / 12
        config_choice["opt_params"]["learning_rate_main"] = val
        return config_choice["opt_type"](**config_choice["opt_params"])

    config_accelerated_lorentz_centroid = {
        "config_id": 1120,
        "name": "Octree Lorentz Centroid",
        "data_num_components": 3,
        "opt_type": SequentialOptimizer.sequence_lorentz_proj,
        "opt_params": {
            "learning_rate_ex": None,
            "learning_rate_main": None,
            "exaggeration": 12,
            "vanilla": False,
            "momentum_ex": 0.35,
            "momentum": 0.6,
            "exact": False,
            "lorentz_centroid": True,
            "n_iter_check": 10,
            "hyperbolic_model": "lorentz",
            "size_tol": 0.96,
        },
        "get_opt_params": config_accelerated_lorentz_centroid_get_opt_params,
    }


def find_config(ci, cfg):
    for c in ci.get_all_configs():
        if isinstance(cfg, str) and c["name"] == cfg:
            return c
        elif isinstance(cfg, int) and c["config_id"] == cfg:
            return c


def load_vars_env():
    load_dotenv()

    return {
        "results_path": os.getenv('RESULTS_DIR'),
        "datasets_path": os.getenv('DATASETS_DIR'),
    }


def setup_experiment(cfgs_list):
    if not isinstance(cfgs_list, list):
        raise ValueError("Configs parameter has to be a list!")

    cfgs = []
    ci = TSNEConfigs()
    for cfg in cfgs_list:
        cfgs.append(find_config(ci, cfg))

    return ci, cfgs, load_vars_env()
