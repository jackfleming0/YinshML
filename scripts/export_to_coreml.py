import argparse
import torch
import coremltools as ct
from collections import namedtuple
import os
"""
Usage looks like this:
from project root
python scripts/export_to_coreml.py /checkpoints/combined/separate_value_head_2/checkpoint_iteration_13.pt /checkpoints/combined/separate_value_head_2/checkpoint_iteration_13.mlpackage

"""



# Correct imports based on your project structure
from yinsh_ml.network.wrapper import NetworkWrapper, StateEncoder, ModelOutput

def export_experiment_to_coreml(pt_model_path: str, coreml_output_path: str):
    """
    Exports a PyTorch model from an experimental run to CoreML format.

    Args:
        pt_model_path: Path to the PyTorch model (.pt file).
        coreml_output_path: Path to save the exported CoreML model.
    """
    # Create a NetworkWrapper instance
    wrapper = NetworkWrapper()

    # Load the PyTorch model
    wrapper.load_model(pt_model_path)

    # Export the model to CoreML
    wrapper.export_to_coreml(coreml_output_path)
    print(f"Successfully exported {pt_model_path} to {coreml_output_path}")

def main():
    """
    Parses command-line arguments and exports a PyTorch model to CoreML.
    """
    parser = argparse.ArgumentParser(description="Export a PyTorch Yinsh model to CoreML format.")
    parser.add_argument("pt_model_path", help="Path to the PyTorch model (.pt file).")
    parser.add_argument("coreml_output_path", help="Path to save the exported CoreML model (.mlpackage).")

    args = parser.parse_args()

    export_experiment_to_coreml(args.pt_model_path, args.coreml_output_path)

if __name__ == "__main__":
    main()