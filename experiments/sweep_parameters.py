import numpy as np
import argparse
import os
import yaml
from src import agent

def convert(argument):
    return list(map(int, argument.split(',')))

parser = argparse.ArgumentParser('')
parser.add_argument('--gpu_idx', type=int, default=0)
parser.add_argument('--n_networks', type=int, default=1)
parser.add_argument('--n_training_iterations', type=int, default=175)
parser.add_argument('--n_evaluation_iterations', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_stim_batches', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.02)
parser.add_argument('--top_down_grad_multiplier', type=float, default=0.1)
parser.add_argument('--clip_grad_norm', type=float, default=1.)
parser.add_argument('--adam_epsilon', type=float, default=1e-7)
parser.add_argument('--max_h_for_output', type=float, default=5.)
parser.add_argument('--n_learning_rate_ramp', type=int, default=20)
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--steady_state_start', type=float, default=1300)
parser.add_argument('--steady_state_end', type=float, default=1700)
parser.add_argument('--training_type', type=str, default='supervised')
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/good_params.yaml')
parser.add_argument('--params_range_fn', type=str, default='./rnn_params/param_ranges.yaml')
parser.add_argument('--save_path', type=str, default=f'./results/new_tasks/param_sweeps/')
parser.add_argument('--restrict_output_to_exc', type=bool, default=False)
parser.add_argument('--task_set', type=str, default='2stim')
parser.add_argument('--model_type', type=str, default='model_experimental')
parser.add_argument('--n_RFs', type=int, default=2)
parser.add_argument('--top_down_trainable', type=util.str2bool, default=True)
parser.add_argument('--readout_trainable', type=util.str2bool, default=True, nargs="?")
parser.add_argument('--aggressive', type=bool, default=True)
parser.add_argument('--min_sample_decode', type=float, default=0.75)
parser.add_argument('--bottom_up_topology', type=bool, default=True)
parser.add_argument('--n_top_down_hidden', type=int, default=64)
parser.add_argument('--training_alg', type=str, default='BPTT')


if __name__ == "__main__":

    args = parser.parse_args()

    # If location for saving results doesn't yet exist, set it up
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Load in basic params + ranges
    rnn_params = yaml.load(open(args.rnn_params_fn), Loader=yaml.FullLoader)
    param_ranges = yaml.load(open(args.params_range_fn), Loader=yaml.FullLoader)
    rnn_params = argparse.Namespace(**rnn_params)

    # Define the agent
    agent = agent.Agent(args, rnn_params, param_ranges)
    
    full_runs = 0
    for i in range(100000000):
        print(f'Main loop iteration {i} - Full runs {full_runs}')
        params =  {k:v for k,v in vars(self._rnn_params).items()}
        for k, v in param_ranges.items():
            if v[0] == v[1]:
                new_value = v[0]
            else:
                new_value = np.random.uniform(v[0], v[1])
            params[k] = new_value


        success = agent.sweep(argparse.Namespace(**params), i)
        if success:
            full_runs += 1
