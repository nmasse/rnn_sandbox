import os, pickle, scipy, argparse, yaml, copy, numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser('')
parser.add_argument('data_dir', type=str, default='./results/')
parser.add_argument('--base_dir', type=str, default='/home/mattrosen/rnn_sandbox/')
parser.add_argument('--yaml_save_path', type=str, default='/Users/mattrosen/rnn_sandbox/rnn_params/reliability_test/')
parser.add_argument('--npz_save_path', type=str, default='/Users/mattrosen/param_data/')
parser.add_argument('--yaml_save_path_control', type=str, default='/Users/mattrosen/rnn_sandbox/rnn_params/control_params_for_RL/')
parser.add_argument('--n_folds', type=int, default=14)
parser.add_argument('--save_out', type=bool, default=False)
parser.add_argument('--save_out_control', type=bool, default=False)

if __name__ == "__main__":

    # Parse arguments, then look through all params
    args = parser.parse_args()

    # Set up records, as well as save location
    if not os.path.exists(args.yaml_save_path):
        os.makedirs(args.yaml_save_path)
    if not os.path.exists(args.npz_save_path):
        os.makedirs(args.npz_save_path)
    if not os.path.exists(args.yaml_save_path_control) and args.save_out_control:
        os.makedirs(args.yaml_save_path_control)

    accs           = []
    params         = []
    initial_mean_h = []
    min_accs       = []
    mean_accs      = []
    median_accs    = []
    filenames      = []

    # Loop through all .pkl files containing results; record acc/params/fn
    fns = os.listdir(args.data_dir)
    for fn in fns:
        f = os.path.join(args.data_dir, fn)
        if os.path.isdir(f) or not f.endswith(".pkl"):
            continue
        x = pickle.load(open(f,'rb'))

        if len(x['task_accuracy']) > 0:

            # Prepare params
            p = vars(x['rnn_params'])
            for k, v in p.items():
                if type(v) == np.int64:
                    p[k] = int(v)
                if type(v) == str:
                    p[k] = float(v)

            p['filename'] = fn

            # Record results
            initial_mean_h.append(x['initial_mean_h'][15:])
            task_acc = np.array(x['task_accuracy'])
            min_accs.append(np.amin(task_acc[-25:, :].mean(axis=0)))
            mean_accs.append(task_acc[-25:, :].mean())
            filenames.append(fn)
            params.append(p)

    # Sort all arrays in order according to accuracy
    order = np.argsort(mean_accs)[::-1]
    for i, k in enumerate(order):
        fold = i % args.n_folds
        n_within_fold = i // args.n_folds
        acc = mean_accs[k]
        p = params[k]
        fr = initial_mean_h[k]

        fn_prefix = f"fold={fold}_number={n_within_fold}_acc={acc:.4f}"

        # Save out results: YAMLs
        if i < 400 and args.save_out:
            save_loc_yaml = os.path.join(args.yaml_save_path, fn_prefix + ".yaml")
            with open(save_loc_yaml, 'w') as outfile:
                yaml.dump(p, outfile, default_flow_style=False)
            
        # Save out results: NPZs
        np.savez_compressed(os.path.join(args.npz_save_path, fn_prefix + ".npz"), 
            params=list(p.values()), 
            keys=list(p.keys()), 
            acc=acc,
            fr=fr)

    # Now: for all param sets w/ < 0.55 accuracy, sort in order of firing rate,
    # and save 8 with low max and low mean
    mean_init_h = np.array(initial_mean_h).mean(1)
    max_init_h = np.array(initial_mean_h).max(1)
    mean_accs = np.array(mean_accs)
    lo_acc = np.where(mean_accs < 0.55)
    acceptable_fr_mean = np.where(mean_init_h[lo_acc] > 0.05)[0]
    acceptable_fr_max = np.where(max_init_h[lo_acc] < 10)[0]
    acceptable_fr = np.intersect1d(acceptable_fr_mean, acceptable_fr_max)
    top_8 = np.argsort(mean_init_h[lo_acc][acceptable_fr])[:8]
    params_arr = np.array(params)

    for i,o in enumerate(top_8):
        # Print out key info
        print(mean_accs[lo_acc][acceptable_fr][o], 
            mean_init_h[lo_acc][acceptable_fr][o], 
            max_init_h[lo_acc][acceptable_fr][o], 
            params_arr[lo_acc][acceptable_fr][o]['tc_modulator'])
        p = params_arr[lo_acc][acceptable_fr][o]

        # Save params for these nets
        save_loc_yaml = os.path.join(args.yaml_save_path_control, 
            f"control_number={i}_acc={mean_accs[lo_acc][acceptable_fr][o]}.yaml")
        with open(save_loc_yaml, 'w') as outfile:
            yaml.dump(p, outfile, default_flow_style=False)
