import numpy as np, os, glob, pickle, argparse, yaml, scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as mcolors
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

parser = argparse.ArgumentParser('')
parser.add_argument('data_path_rel', type=str)
parser.add_argument('data_path_gen', type=str)
parser.add_argument('--figure_save_path', type=str, default='../reliability_figures/')
parser.add_argument('--yaml_save_path', type=str, default='./rnn_params/reliable_params_for_RL/')
parser.add_argument('--save_out', type=bool, default=False)

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.exists(args.yaml_save_path):
        os.makedirs(args.yaml_save_path)

    # Load reliability data
    frs = []
    mean_accs = []
    min_accs = []
    orig_accs = []
    filenames = []
    params = []
    tc_mods_rel = []
    for fn in glob.glob(os.path.join(args.data_path_rel, "*.pkl")):
        data_all = pickle.load(open(fn, 'rb'))
        data = np.array(data_all['task_accuracy'])[:,-25:,:]
        acc = np.mean(data, axis=(1,2))
        mean_accs.append(acc)
        min_accs.append(np.amin(np.mean(data, axis=1),axis=1))
        frs.append(np.mean(data_all['initial_mean_h']))
        orig_accs.append(data_all['original_task_accuracy'])
        filenames.append(os.path.basename(fn))
        params.append(vars(data_all['rnn_params']))
        tc_mods_rel.append(data_all['rnn_params'].tc_modulator)

    # Load generalizability data
    gen_accs = []
    tc_mods_gen = []
    for fn in glob.glob(os.path.join(args.data_path_gen, "*.pkl")):
        data_all = pickle.load(open(fn, 'rb'))
        data = np.array(data_all['task_accuracy'])[:,-25:,:]
        acc = np.mean(data, axis=(1,2))
        gen_accs.append(acc)
        tc_mods_gen.append(data_all['rnn_params'].tc_modulator)

    # Compute mean/std dev of accuracy
    min_acc  = np.amax(np.array(min_accs), axis=1)
    mean_acc = np.mean(np.array(mean_accs), axis=1)#np.median(np.hstack((np.array(orig_accs)[:,np.newaxis],np.array(mean_accs))), axis=1)#
    sd_acc   = np.std(np.hstack((np.array(orig_accs)[:,np.newaxis],np.array(mean_accs))), axis=1)
    mean_gen_acc = np.mean(np.array(gen_accs), axis=1)

    # Make reliability accuracy scatterplot: 
    # (a) sd acc of repetitions as fxn of orig acc
    # (b) sd acc of repetitions as fxn of mean acc of repetitions
    # (c) joint distribution of mean acc/sd acc
    # (d) scatterplot of sd acc vs. mean firing rate
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    ax[0,0].scatter(orig_accs, sd_acc)
    ax[0,0].set(xlabel="Original acc.", ylabel="$\sigma$ repetition acc.")
    ax[0,1].scatter(mean_acc, sd_acc)
    ax[0,1].set(xlabel="Mean repetition acc.", ylabel="$\sigma$ repetition acc.")
    _,_,_,c = ax[1,0].hist2d(mean_acc, sd_acc, bins=(20,20), density=True)
    ax[1,0].set(xlabel="Mean repetition acc.", ylabel="$\sigma$ repetition acc.")
    fig.colorbar(c, ax=ax[1,0])
    ax[1,1].scatter(frs, sd_acc)
    ax[1,1].set(xlabel="Mean pre-training firing rate", ylabel="$\sigma$ repetition acc.")
    plt.tight_layout()
    fig.savefig(os.path.join(args.figure_save_path, "reliability_acc.png"), dpi=500)
    plt.close(fig)

    fig, ax = plt.subplots(1)
    most_reliable = np.where((sd_acc < 2e-2) & (mean_acc > 0.8))[0]
    
    print(np.array(frs)[most_reliable])
    print(len(frs), len(sd_acc), len(min_acc))
    ax.scatter(np.repeat(np.array(orig_accs)[most_reliable],5), np.array(mean_accs)[most_reliable].flatten())
    ax.set(xlabel="Original acc.", ylabel="$\sigma$ repetition acc.")
    fig.savefig(os.path.join(args.figure_save_path, "most_reliable_params.png"), dpi=500)
    plt.close(fig)

    fig, ax = plt.subplots(1)
    ax.scatter(np.array(orig_accs), mean_acc)
    ax.set(xlim=[0.5, 1.0], ylim=[0.5, 1.0])
    ax.plot(np.linspace(0.5, 1.0, 1000), np.linspace(0.5, 1.0, 1000))
    ax.set(xlabel="Original accuracy", ylabel="Mean repetition accuracy")
    plt.tight_layout()
    fig.savefig(os.path.join(args.figure_save_path, "new_vs_original_accuracy.png"), dpi=500)
    plt.close(fig)

    # Plot generalizability vs. reliability
    fig, ax = plt.subplots(1)
    ordering_tc_gen = np.argsort(tc_mods_gen)
    ordering_tc_rel = np.argsort(tc_mods_rel)
    ax.scatter(mean_acc[ordering_tc_rel], mean_gen_acc[ordering_tc_gen])
    ax.plot(np.linspace(0.2, 1.0, 1000), np.linspace(0.2, 1.0, 1000))
    ax.set(xlabel="Mean accuracy (original 7 tasks)", ylabel="Mean accuracy (challenge tasks)")
    plt.tight_layout()
    fig.savefig(os.path.join(args.figure_save_path, "challenge_vs_original_accuracy.png"), dpi=500)

    # Full three-part plot: 
    # (a) mean vs std dev of repetitions; 
    # (b) zoom-in: mean accuracy for nets > 0.85 vs. std dev for those nets
    # (c) generalizability: mean accuracy on challenge vs mean accuracy on original
    slope, intercept, r, p, se = scipy.stats.linregress(mean_acc[ordering_tc_rel], mean_gen_acc[ordering_tc_gen])
    
    corr = scipy.stats.pearsonr(mean_acc[ordering_tc_rel], mean_gen_acc[ordering_tc_gen])[0]
    print(r**2, slope, corr)
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    h2d,xbins,ybins = np.histogram2d(mean_acc, sd_acc)
    cmap = plt.get_cmap('viridis', int(np.max(h2d)-np.min(h2d)+1))
    _,_,_,c = ax[0].hist2d(mean_acc, sd_acc, bins=[xbins,ybins],cmin=h2d.min()-0.5, cmax=h2d.max()+0.5,cmap=cmap)
    ax[0].set(xlabel="Mean repetition accuracy", ylabel="$\sigma$ repetition accuracy")
    fig.colorbar(c, ax=ax[0], ticks=np.arange(h2d.min(),h2d.max()+1,5))

    h2d,xbins,ybins = np.histogram2d(mean_acc[mean_acc > 0.85], sd_acc[mean_acc > 0.85])
    most_reliable = np.where((mean_acc >= xbins[-4]) & (sd_acc <= ybins[-3]))[0]
    for i, j,sd, k in zip(np.array(frs)[most_reliable], mean_acc[most_reliable], sd_acc[most_reliable], most_reliable):
        #print(f"{i:.2f},{j:.2f},{sd:.2f},{filenames[k]}, {params[k]['tc_modulator']}")
        if i > 0.1:
            print(f"{params[k]['tc_modulator']},")
    print()
    for i, j,sd, k in zip(np.array(frs)[most_reliable], mean_acc[most_reliable], sd_acc[most_reliable], most_reliable):
        #print(f"{i:.2f},{j:.2f},{sd:.2f},{filenames[k]}, {params[k]['tc_modulator']}")
        if i < 0.1:
            print(f"{params[k]['tc_modulator']},")
    ncolors = int(h2d.max() - h2d.min() + 1)
    cmap = plt.get_cmap("viridis")
    _,_,_,c = ax[1].hist2d(mean_acc[mean_acc > 0.85], sd_acc[mean_acc > 0.85], bins=[xbins,ybins],cmap=cmap)
    ax[1].set(xlabel="Mean repetition accuracy", ylabel="$\sigma$ repetition accuracy")
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    f = fig.colorbar(mappable, ax=ax[1])
    f.set_ticks(np.linspace(0, ncolors, ncolors))
    f.set_ticklabels(range(ncolors))

    #_,_,_,c1 = ax[1].hist2d(mean_acc[mean_acc > 0.85], sd_acc[mean_acc > 0.85])
    #ax[1].set(xlabel="Mean repetition accuracy (> 0.85)", ylabel="$\sigma$ repetition accuracy",)
    #fig.colorbar(c1, ax=ax[1])
    ax[2].scatter(mean_acc[ordering_tc_rel], mean_gen_acc[ordering_tc_gen])
    x = np.linspace(mean_acc.min(), mean_acc.max(), 1000)
    #ax[2].plot(x, intercept + slope * x , 'r')
    ax[2].set(xlim=[0.4,1.0], ylim=[0.4,1.0],xlabel="Mean accuracy (original tasks)", ylabel="Mean accuracy (challenge tasks)")
    plt.tight_layout()
    fig.savefig(os.path.join(args.figure_save_path, "reliability_generalizability_full.png"), dpi=500)

    for i in most_reliable:
        print(np.array(frs)[i], params['tc_modulator'])

    # ADD IN LOGIC HERE FOR SELECTING NETS BASED ON RELIABILITY + ACCURACY
    if args.save_out:
        for i in most_reliable:
            parameter_fn = os.path.splitext(filenames[i])[0] + ".yaml"
            save_loc_yaml = os.path.join(args.yaml_save_path, parameter_fn)
            for k, v in params[i].items():
                if type(v) == np.int64:
                    params[i][k] = int(v)
                if type(v) == str:
                    params[i][k] = float(v)
            
            with open(save_loc_yaml, 'w') as outfile:
                yaml.dump(params[i], outfile, default_flow_style=False)