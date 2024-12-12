import datetime  # Add this with other imports
import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

def setup_distributed():
    """Initialize distributed training"""
    try:
        # Get SLURM variables
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # Print debug info
        print(f"Process {rank}: CUDA devices: {torch.cuda.device_count()}")
        print(f"Process {rank}: Local rank: {local_rank}")
        
        # Initialize distributed process group first
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Then set device after process group is initialized
        '''
        if torch.cuda.is_available():
            gpu_id = local_rank  # Assuming LOCAL_RANK matches GPU index in CUDA_VISIBLE_DEVICES
            torch.cuda.set_device(gpu_id)
            print(f"Process {rank}: Mapped to GPU {gpu_id}")
        '''
            
        return rank, local_rank, world_size
        
    except Exception as e:
        print(f"Error in setup_distributed: {str(e)}")
        raise e

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default="pmub_linear.yml", help="Path to the config file"
    )
    parser.add_argument(
        "--dataset", type=str, default="PMUB", help="Name of dataset(LDFDCT, BRATS, PMUB)"
    )
    parser.add_argument("--seed", type=int, default=1244, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        default="Fast-DDPM_experiments",
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_false",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="uniform",
        help="sample involved time steps according to (uniform or non-uniform)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
    
    parser.add_argument('--distributed', action='store_true', 
                       help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    
    
    args = parser.parse_args()
    rank = int(os.environ.get('SLURM_PROCID', 0))
    is_main_process = rank == 0
    
    # Modify paths to include rank for distributed training
    if args.distributed:
        args.doc = f"{args.doc}_rank{rank}"
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # Only the main process should handle directory cleanup/creation
    if not args.test and not args.sample:
        if not args.resume_training:
            # Create directories safely
            os.makedirs(args.log_path, exist_ok=True)
            os.makedirs(tb_path, exist_ok=True)
            
            if is_main_process:  # Only main process handles overwrites
                if os.path.exists(args.log_path):
                    if args.ni:
                        try:
                            shutil.rmtree(args.log_path)
                            if os.path.exists(tb_path):
                                shutil.rmtree(tb_path)
                        except FileNotFoundError:
                            pass
                    else:
                        response = input(f"Folder {args.log_path} exists. Overwrite? (Y/N)")
                        if response.upper() == "Y" or (1 == 1):
                            try:
                                shutil.rmtree(args.log_path)
                                if os.path.exists(tb_path):
                                    shutil.rmtree(tb_path)
                            except FileNotFoundError:
                                pass

            # Recreate directories
            os.makedirs(args.log_path, exist_ok=True)
            os.makedirs(tb_path, exist_ok=True)

            # Save config
            if is_main_process:
                with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                    yaml.dump(new_config, f, default_flow_style=False)

    # Set up tensorboard for non-test, non-sample runs
    if not args.test and not args.sample and not args.resume_training:
        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)

    # Setup logging
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    # File logging only for main process
    if is_main_process:
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        handler2.setFormatter(formatter)
        logger.addHandler(handler2)

    # Device setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if is_main_process:
        logging.info("Using device: {}".format(device))
    new_config.device = device

    # Random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    
    # Set up distributed training if enabled

    try:
        runner = Diffusion(args, config)
        
        
        if args.sample:
            print("IM IN SAMPLE")
            if args.dataset=='PMUB':
                runner.sr_sample()
            elif args.dataset=='LDFDCT' or args.dataset=='BRATS' or args.dataset=='CheXpert':
                runner.sg_sample()
            else:
                raise Exception("This script only supports LDFDCT, BRATS and PMUB as sampling dataset.")
        elif args.test:
            runner.test()
        else:
            print("IM IN TRAIN")
            rank, local_rank, world_size = setup_distributed()
            runner.rank = rank
            runner.local_rank = local_rank
            runner.world_size = world_size

                # Only log from rank 0
            if rank == 0:
                logging.info("Writing log file to {}".format(args.log_path))
                logging.info("Exp instance id = {}".format(os.getpid()))
                logging.info("Exp comment = {}".format(args.comment))
                logging.info(f"Initialized distributed training: rank {rank}/{world_size}")

            if args.dataset=='PMUB':
                runner.sr_train()
            elif args.dataset=='LDFDCT' or args.dataset=='BRATS' or args.dataset=='CheXpert':
                runner.sg_train()
            else:
                raise Exception("This script only supports LDFDCT, BRATS and PMUB as training dataset.")
    except Exception:
        logging.error(traceback.format_exc())
    finally:
        # Clean up distributed training
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
