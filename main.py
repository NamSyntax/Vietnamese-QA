import argparse
import sys
import yaml

from src.train import run_train
from src.evaluate import run_evaluate
from src.inference import run_gradio

def main():
    """
    Main entry point for the project.
    Parses command-line arguments to route execution: train, eval, or deploy (Gradio/FastAPI).
    """
    parser = argparse.ArgumentParser(description="Vietnamese QA - MLOps Entrypoint")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval", "gradio", "api"], 
                        help="Execution mode: train, eval, gradio, or api")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to the YAML configuration file")
    
    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
    
    # Route execution flow based on the provided mode
    if args.mode == "train":
        print("Starting model training process ...")
        run_train(config)
    elif args.mode == "eval":
        print("Starting model evaluation ...")
        run_evaluate(config)
    elif args.mode == "gradio":
        print("Launching Gradio demo application ...")
        run_gradio(config)
    elif args.mode == "api":
        print("Starting FastAPI server ...")
        from src.api import run_api
        run_api(config)

if __name__ == "__main__":
    main()

