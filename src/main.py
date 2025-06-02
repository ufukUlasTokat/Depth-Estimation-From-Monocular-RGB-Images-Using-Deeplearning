import argparse
import torch

from train import train_model
from test import test_model

def main():
    parser = argparse.ArgumentParser(description="Depth Estimation Trainer/Tester")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Choose 'train' to train the model or 'test' to evaluate it.")
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet34", "densenet", "mobilenet", "efficientnet"],
                        required=True, help="Model architecture to use.")
    parser.add_argument("--dataset", type=str, choices=["sunrgbd", "nyudepth"], required=True,
                        help="Dataset to use.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model weights (required for test mode).")
    parser.add_argument("--visualize_index", type=int, default=None,
                        help="Optional index for visualization in test mode.")

    args = parser.parse_args()

    if args.mode == "train":
        print(f"🚀 Training {args.model} on {args.dataset}")
        train_model(args.model, args.dataset)

    elif args.mode == "test":
        if not args.model_path:
            raise ValueError("You must specify --model_path in test mode.")
        print(f"🧪 Testing {args.model} on {args.dataset}")
        test_model(model_name=args.model,
                   dataset_name=args.dataset,
                   model_path=args.model_path,
                   visualize_index=args.visualize_index)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
