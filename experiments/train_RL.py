import tensorflow as tf
import numpy as np
import argparse, pickle, os, yaml, glob
import matplotlib.pyplot as plt
from src import util
from src.RLagent import RLAgent

parser = argparse.ArgumentParser('')
parser.add_argument('--gpu_idx', type=int, default=0)
parser.add_argument('--n_networks', type=int, default=1)
parser.add_argument('--n_training_episodes', type=int, default=2000)
parser.add_argument('--n_evaluation_episodes', type=int, default=50)
parser.add_argument('--n_episodes', type=int, default=2000)
parser.add_argument('--time_horizon', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--n_minibatches', type=int, default=4)
parser.add_argument('--clip_ratio', type=float, default=0.1)
parser.add_argument('--normalize_gae', type=util.str2bool, default=False)
parser.add_argument('--normalize_gae_cont', type=util.str2bool, default=False)
parser.add_argument('--entropy_coeff', type=float, default=0.002)
parser.add_argument('--critic_coeff', type=float, default=1.)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--cont_learning_rate', type=float, default=5e-6)
parser.add_argument('--n_learning_rate_ramp', type=int, default=20)
parser.add_argument('--decay_learning_rate', type=util.str2bool, default=True)
parser.add_argument('--clip_grad_norm', type=float, default=1.)
parser.add_argument('--clip_grad_norm_cont', type=float, default=0.5)
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--training_type', type=str, default='RL')
parser.add_argument('--rnn_params_path', type=str, default='./rnn_params/new_tasks/')
parser.add_argument('--rnn_params_fn', type=str, default='full_2.yaml')
parser.add_argument('--save_path', type=str, default='./results/RL/reliable_params_cont_norm/')
parser.add_argument('--start_action_std', type=float, default=0.1)
parser.add_argument('--end_action_std', type=float, default=0.01)
parser.add_argument('--OU_noise', type=util.str2bool, default=False)
parser.add_argument('--OU_theta', type=float, default=0.15)
parser.add_argument('--OU_clip_noise', type=float, default=3.)
parser.add_argument('--max_h_for_output', type=float, default=5.)
parser.add_argument('--action_bound', type=float, default=5.)
parser.add_argument('--cont_action_dim', type=int, default=64)
parser.add_argument('--disable_cont_action', type=bool, default=False)
parser.add_argument('--restrict_output_to_exc', type=bool, default=False)
parser.add_argument('--cont_action_multiplier', type=float, default=1.0)
parser.add_argument('--model_type', type=str, default='model_experimental')
parser.add_argument('--task_set', type=str, default='2stim')
parser.add_argument('--steady_state_start', type=int, default=1300)
parser.add_argument('--steady_state_end', type=int, default=1700)
parser.add_argument('--do_overwrite', type=util.str2bool, default=False, nargs="?")
parser.add_argument('--top_down_trainable', type=util.str2bool, default=True, nargs="?")
parser.add_argument('--bottom_up_topology', type=util.str2bool, default=True, nargs="?")
parser.add_argument('--top_down_overlapping', type=util.str2bool, default=True, nargs="?")
parser.add_argument('--motion_restriction', type=util.str2bool, default=True, nargs="?")
parser.add_argument('--readout_trainable', type=util.str2bool, default=True, nargs="?")
parser.add_argument('--activity_cap', type=float, default=5.0)
parser.add_argument('--verbose', type=util.str2bool, default=True)
parser.add_argument('--training_alg', type=str, default='BPTT')


if __name__ == "__main__":

    args = parser.parse_args()

    # Set up for saving 
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Load in filename for parameters to run
    rp = os.path.join(args.rnn_params_path, args.rnn_params_fn)

    # Check if file already exists in save directory
    save_fn = os.path.join(args.save_path, os.path.basename(rp))
    mean_accuracy = os.path.basename(rp)

    # Check whether to re-run or not
    if os.path.exists(save_fn[:-5] + ".pkl") and not args.do_overwrite:
        print(save_fn + " completed!")
        exit()

    # Lock this so another process doesn't overwrite
    pickle.dump({}, open(save_fn[:-5] + ".pkl", 'wb'))
    start = mean_accuracy.find("acc=")+4
    end = start + 6
    try:
        mean_accuracy = float(mean_accuracy[start:end])
    except ValueError:
        mean_accuracy = None

    # Load in parameter set 
    rnn_params = yaml.load(open(rp), Loader=yaml.FullLoader)
    args.rnn_params_fn = rp
    rnn_params = argparse.Namespace(**rnn_params)
    agent = RLAgent(args, rnn_params)

    for n in range(args.n_networks):
        fn = f"_rep={n}".join(os.path.splitext(os.path.basename(rp))
        agent.train(
            rnn_params,
            fn,
            n)
