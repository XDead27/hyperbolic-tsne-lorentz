"""
This file contains different configurations for experiments to be run on.
"""

from hyperbolicTSNE import SequentialOptimizer

class TSNEConfigs:

    def config_accelerated_poincare_get_opt_params(self, n_samples):
        val = n_samples / (self.config_accelerated_poincare["opt_params"]["exaggeration"] * 1000)

        self.config_accelerated_poincare["opt_params"]["learning_rate_ex"] = val
        self.config_accelerated_poincare["opt_params"]["learning_rate_main"] = val
        return self.config_accelerated_poincare["opt_type"](**self.config_accelerated_poincare["opt_params"])

    config_accelerated_poincare = {
        "config_id": 1000,
        "name": "Quadtree Poincare",
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
    Lorentz Hyperboloid Configuration
    """
    def config_accelerated_lorentz_get_opt_params(self, n_samples):
        val = n_samples / (self.config_accelerated_lorentz["opt_params"]["exaggeration"] * 5)

        self.config_accelerated_lorentz["opt_params"]["learning_rate_ex"] = val / 12
        self.config_accelerated_lorentz["opt_params"]["learning_rate_main"] = val
        return self.config_accelerated_lorentz["opt_type"](**self.config_accelerated_lorentz["opt_params"])

    config_accelerated_lorentz = {
        "config_id": 1100,
        "name": "Octree Lorentz",
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
        "get_opt_params": config_accelerated_poincare_get_opt_params,
    }
