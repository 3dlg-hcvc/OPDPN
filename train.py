import argparse
from time import time
import logging
import numpy as np 
import torch
import random
import os
from PC_lib import PCTrainer

def get_parser():
    parser = argparse.ArgumentParser(description="Train motion net")
    parser.add_argument(
        "--test",
        action="store_true",
        help="indicating whether to only do evaluation",
    )
    parser.add_argument(
        "--train_path",
        default=None,
        metavar="FILE",
        help="hdf5 file which contains the train data",
    )
    parser.add_argument(
        "--test_path",
        required=True,
        metavar="FILE",
        help="hdf5 file which contains the test data",
    )
    parser.add_argument(
        "--inference_model",
        default=None,
        metavar="FILE",
        help="the inference file when test is True",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        metavar="DIR",
        help="hdf5 file which contains the test data",
    )
    parser.add_argument(
        "--max_K",
        default=5,
        type=int,
        help="indicatet the max number for the segmentation",
    )
    parser.add_argument(
        "--category_number",
        default=3,
        type=int,
        help="indicate the number of part categories",
    )
    # Some default hyperparameters
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="choose from cuda or cpu",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_points",
        default=1024,
        help="Number of points used to sample the input data"
    )
    parser.add_argument(
        "--random_seed",
        default=42,
    )
    
    return parser

if __name__ == "__main__":
    start = time()

    args = get_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger('train')
    log.info("Arguments: " + str(args))
    
    # Setup some default value
    args.data_path = {"train": args.train_path, "test": args.test_path}
    args.save_frequency = 100
    args.log_frequency = 10
    
    args.loss_weight = {
        "loss_category": 1.0,
        "loss_instance": 1.0,
        "loss_mtype": 1.0,
        "loss_maxis": 1.0,
        "loss_morigin": 1.0,
    }

    # Make the training deterministic
    seed = args.random_seed
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    trainer = PCTrainer(args, args.max_K, args.category_number)
    if not args.test:
        log.info(f'Train on {args.train_path}, validate on {args.test_path}')
        trainer.train()
    else:
        log.info(f'Test on {args.test_path} with inference model {args.inference_model}')
        trainer.test(inference_model=args.inference_model)

    stop = time()
    log.info(f"Total time: {stop-start}")
