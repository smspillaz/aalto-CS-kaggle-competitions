from skorch.callbacks import Callback

class SummarizeParameters(Callback):
    """Summarize parameters at the beginning of training."""
    
    def on_train_begin(self, net, X, y):
        """Print out some useful information."""
        super().on_train_begin(net, X, y)
        
        PRETTY_MAP = {
            "Module": "module",
            "Optimizer": "optimizer",
            "Criterion": "criterion",
        }
        PRETTY_PARAMS = set(PRETTY_MAP.values())
        RESERVED_PARAMS = set([
            "callbacks",
            "iterator_train",
            "iterator_valid",
            "dataset",
            "train_split",
            "history",
            "warm_start",
        ])
        net_params = dict([
            (k, v) for k, v in net.get_params(deep=False).items()
            if not k.endswith("_") and k not in RESERVED_PARAMS
        ])

        print("Training {}".format(net.__class__.__name__))
        for pretty_name, prefix in PRETTY_MAP.items():
            print(" - {}".format(pretty_name))
            for attr, _ in net_params.items():
                if attr.startswith(prefix):
                    print("   - {}:".format(attr))
        
        # Now print all the remaining net parameters
        print(" - Remaining Paramters")
        for attr, value in net_params.items():
            if not any([attr.startswith(p) for p in PRETTY_PARAMS]):
                print("    - {}: {}".format(attr, value))

