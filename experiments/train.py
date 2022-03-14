import numpy as np
import argparse
import pickle
import os
import yaml
import glob
from src import util
from src.agent import Agent


parser = argparse.ArgumentParser('')
parser.add_argument('--gpu_idx', type=int, default=0)
parser.add_argument('--n_networks', type=int, default=1)
parser.add_argument('--n_training_iterations', type=int, default=175)
parser.add_argument('--n_evaluation_iterations', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_stim_batches', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.02)
parser.add_argument('--adam_epsilon', type=float, default=1e-7)
parser.add_argument('--clip_grad_norm', type=float, default=1.)
parser.add_argument('--top_down_grad_multiplier', type=float, default=0.1)
parser.add_argument('--n_learning_rate_ramp', type=int, default=20)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/good_params.yaml')
parser.add_argument('--params_folder', type=str, default='./rnn_params/new_tasks/')
parser.add_argument('--training_type', type=str, default='supervised')
parser.add_argument('--save_path', type=str, default='./results/reliability_assessment/')
parser.add_argument('--max_h_for_output', type=float, default=5.)
parser.add_argument('--steady_state_start', type=int, default=1300)
parser.add_argument('--steady_state_end', type=int, default=1700)
parser.add_argument('--task_set', type=str, default='2stim')
parser.add_argument('--model_type', type=str, default='model_experimental')
parser.add_argument('--restrict_output_to_exc', type=bool, default=False)
parser.add_argument('--do_overwrite', type=util.str2bool, default=False, nargs="?")
parser.add_argument('--top_down_trainable', type=util.str2bool, default=True, nargs="?")
parser.add_argument('--readout_trainable', type=util.str2bool, default=True, nargs="?")
parser.add_argument('--bottom_up_topology', type=util.str2bool, default=True, nargs="?")
parser.add_argument('--top_down_overlapping', type=util.str2bool, default=True, nargs="?")
parser.add_argument('--save_activities', type=util.str2bool, default=False, nargs="?")
parser.add_argument('--n_top_down_hidden', type=int, default=64)
parser.add_argument('--training_alg', type=str, default='BPTT')


if __name__ == "__main__":

    args = parser.parse_args()

    # Set up for saving 
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Obtain list of parameter files to run 
    params_files = glob.glob(f"{args.params_folder}*.yaml")

    for params_fn in params_files:

        # Check if file already exists in save directory
        save_fn = os.path.join(args.save_path, os.path.basename(params_fn))
        mean_accuracy = os.path.basename(params_fn)

        # Check whether to re-run or not
        if os.path.exists(save_fn[:-5] + ".pkl") and not args.do_overwrite:
            print(save_fn + " completed!")
            continue

        start = mean_accuracy.find("acc=")+4
        end = start + 6
        try:
            mean_accuracy = float(mean_accuracy[start:end])
        except ValueError:
            mean_accuracy = None

        # Load in RNN parameters and train specified n networks
        rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)
        rnn_params = argparse.Namespace(**rnn_params)

        p = yaml.load(open(params_fn), Loader=yaml.FullLoader)
        for k, v in p.items():
            if hasattr(rnn_params, k):
                setattr(rnn_params, k, v)
        agent = Agent(args, rnn_params)
        agent.train(
            rnn_params,
            args.n_networks,
            os.path.basename(params_fn),
            original_task_accuracy=mean_accuracy)
