import argparse

try:
    from simulator.simulator import Simulator
except ImportError:
    import os
    import sys

    SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(SCRIPT_PATH, ".."))
    from simulator.simulator import Simulator


def main(args):
    model_names = ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-70b-hf"]
    simulator = Simulator(model_names=model_names)

    simulator.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default="meta-llama/Llama-2-7b-hf")

    args = parser.parse_args()
    main(args)
