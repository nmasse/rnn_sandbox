import pickle, glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import NearestNeighbors
from scipy import stats, spatial
import warnings, argparse
import os
import scipy.stats

parser = argparse.ArgumentParser('')
parser.add_argument('--data_path', type=str, default='./results/for_activity_analysis/')
parser.add_argument('--figure_save_path', type=str, default='./results/for_activity_analysis/figures/')
parser.add_argument('--n_pcs', type=int, default=7)

warnings.filterwarnings('ignore')
args = parser.parse_args()

def compute_decision_axis_alignment(self, h, rules, outputs, kfolds=4,
        do_plot=True, save_fn=None):
        """ 
        Compute the alignment of the decision axis with activity
        during each task -- specifically, with the regions of the
        state space that correspond with each of the decision
        outputs. Because this is a forced-decision task (e.g.
        only one of the outputs to the exclusion of the others at
        each timepoint), compute the ratio of the projection onto
        one of the output axes vs. the other.
        """
        # Alignment index is given by the ratio of projection
        # onto correct vs. incorrect output (e.g. activity
        # where output 1 is required projected onto axis for output
        # 1 vs. 2, activity where output 2 is required projected
        # onto axis for output 1 vs. 2)

        h_tp = h[:,self.epoch_bounds['test'],:]
        h_tp = h_tp[:,2:,:]
        alignments = np.zeros(self.N_TASKS)
        output_aligned_sep = np.zeros(self.N_TASKS)
        per_trial_alignments = np.zeros((h.shape[0], 2))
        '''
        for i, t in enumerate(np.unique(rules)):
            # Select activity corresponding w/ each task/output
            rel_h_o1 = h_tp[np.where((rules == t) & (outputs == 1))[0]]
            rel_h_o2 = h_tp[np.where((rules == t) & (outputs == 2))[0]]

            # Project activity for each group onto each output + divide
            h_o1_onto_o1 = np.dot(rel_h_o1, self.w_out[:,1]).mean()
            h_o1_onto_o2 = np.dot(rel_h_o1, self.w_out[:,2]).mean()
            h_o2_onto_o1 = np.dot(rel_h_o2, self.w_out[:,1]).mean()
            h_o2_onto_o2 = np.dot(rel_h_o2, self.w_out[:,2]).mean()

            corrects = np.mean([h_o1_onto_o1, h_o2_onto_o2])
            incorrects = np.mean([h_o1_onto_o2, h_o2_onto_o1])
            alignments[i] = corrects / incorrects

            """
            # Other idea: identify the decision axis for the task
            # (Murray-style approach used above), then measure 
            # the angle between this and the output; do so in the 
            # reduced-dimension subspace spanned by decision axes
            """
            # 1. Compute decision axis from activity
            all_data = np.vstack(
                (rel_h_o1.mean(0), rel_h_o2.mean(0)))
            decision_axes_pca = PCA(10).fit(all_data)

            # 2. Compute decision axis in reduced dimension space
            # (project output vectors 1 and 2 onto the 2 axes
            # identified above, then use Murray approach on these
            # coordinates to identify the direct along which variation
            # splits these dimensions)
            w_out_proj_dec_ax = decision_axes_pca.transform(self.w_out[:,1:].T)
            identify_dec_pca = PCA(2).fit(w_out_proj_dec_ax)
            low_d_decision_axis = identify_dec_pca.components_[0,:]

            # 3. Project data into the decision subspace, and measure
            # how well separated activities for different decision types are
            rel_data_all = np.concatenate(
                (np.reshape(rel_h_o1, (-1, self.N_NEUR)),
                np.reshape(rel_h_o2, (-1, self.N_NEUR))),
                axis=0)
            rel_labels = np.concatenate(
                (np.zeros(rel_h_o1.shape[0]*rel_h_o1.shape[1]),
                np.ones(rel_h_o2.shape[0]*rel_h_o2.shape[1])),
                axis=0)
            all_data_low_d = identify_dec_pca.transform(decision_axes_pca.transform(
                rel_data_all))

            all_data_low_d -= np.mean(all_data_low_d, axis=0)

            # 4. Compute percent of data correctly classified based on position
            # wrt y axis
            output_aligned_sep[i] = get_performance(all_data_low_d, rel_labels)
        '''

        """ 
        Newer idea: use the Murray approach to identify the decision axis from activity;
        also use Murray approach to identify decision axis from output weights;
        measure alignment as the amount of variation preserved when projecting in
        both directions.
        """
        w_out_dec_axis = PCA(1).fit(self.w_out[:,1:].T)
        for i, t in enumerate(np.unique(rules)):
            # Select activity corresponding w/ each task/output
            rel_h_o1 = h_tp[np.where((rules == t) & (outputs == 1))[0]]
            rel_h_o2 = h_tp[np.where((rules == t) & (outputs == 2))[0]]
            all_data = np.vstack(
                (rel_h_o1.mean((0,1)), rel_h_o2.mean((0,1))))

            # Fit axis along which cross-output variation exists
            act_dec_axis = PCA(1).fit(all_data)

            # Project onto w_out_dec_axis and act_dec_axis
            h_w_dec_o1 = w_out_dec_axis.transform(np.reshape(rel_h_o1, (-1, self.N_NEUR)))
            h_w_dec_o2 = w_out_dec_axis.transform(np.reshape(rel_h_o2, (-1, self.N_NEUR)))
            h_act_dec_o1 = act_dec_axis.transform(np.reshape(rel_h_o1, (-1, self.N_NEUR)))
            h_act_dec_o2 = act_dec_axis.transform(np.reshape(rel_h_o2, (-1, self.N_NEUR)))

            # Compute amount of separation between output groups for both axes
            dp_w = d_prime(h_w_dec_o1, h_w_dec_o2)
            dp_h = d_prime(h_act_dec_o1, h_act_dec_o2)

            output_aligned_sep[i] = dp_w 
            alignments[i] = dp_h

        print(output_aligned_sep, alignments)
        return alignments#output_aligned_sep

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_fns(data_path=args.data_path):
    return glob.glob(data_path + "*.pkl")

def load_data(fn):

    data = pickle.load(open(fn, 'rb'))
    h = data['final_task_activities'][0]
    weights = data['weights'][0]
    w_td = weights[2] @ weights[4]
    w_rec = weights[5]
    w_out = np.maximum(weights[8], 0.)
    acc = np.mean(np.array(data['task_accuracy'])[:, -25:, :])
    params = data['rnn_params']
    trial_info = data['final_batches'][0]
    print(np.mean(np.array(data['task_accuracy'])[:, -1:, :], axis=1))

    return h, w_td, w_rec, w_out, acc, params, trial_info

def analyze_output_proximity(h, w_td, w_rec, w_out, acc, params, trial_info, fn):

    fn_beg, fn_ext = os.path.splitext(fn)

    # 1st order: What is the overlap between output-projection and topdown-reception?
    # Compute correlation between amount of top-down input received and output selectivity
    topdown_input    = np.sum(w_td, axis=0)
    output_drive     = np.sum(w_out, axis=1)
    output_asymmetry = np.divide(np.amin(w_out[:,1:], axis=1), np.amax(w_out[:,1:],axis=1))
    drive_corr = scipy.stats.spearmanr(topdown_input, output_drive)[0]
    asymm_corr = scipy.stats.spearmanr(topdown_input, output_asymmetry)[0]

    # 2nd order: do hidden units project to outputs in a context-specific way?
    # a. do hidden units receive context specific projections?
    # b. 
    km = KMeans(n_clusters=w_td.shape[0]+1).fit(w_td.T)
    pcs = PCA(2).fit_transform(w_td.T)

    # Sort top-down projections by cluster id, then visualize and save
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,5))
    sorted_w_td = w_td[-7:,km.labels_[np.argsort(km.labels_)]]
    sorted_w_td /= w_td[-7:,:].mean(0)[np.newaxis,:] + 1e-9
    ax[0,0].scatter(topdown_input, output_drive)
    ax[0,1].scatter(topdown_input, output_asymmetry)
    ax[1,0].imshow(sorted_w_td.T, aspect='auto')
    ax[1,1].scatter(pcs[:,0], pcs[:,1], c=np.argmax(w_td, axis=0), cmap=plt.cm.rainbow)
    ax[0,0].set(title=f"TD corr. w/ output drive: {drive_corr:.2f}")
    ax[0,1].set(title=f"TD corr. w/ output asymm: {asymm_corr:.2f}")
    fig.savefig("_".join([fn_beg, "td_clust", fn_ext]), dpi=300)

    # 3. Compute angle between each cue's projection and output readout
    angles_wrt_output = [[angle_between(w_td[i,:], w_out[:,j]) for i in range(w_td.shape[0])] for j in range(w_out.shape[1])]

    # 4. Compute angle between each cue's projection and the top 10 PCs of network activity
    h_pcs = PCA(10).fit(np.reshape(h, (-1, h.shape[-1]))).components_
    angles_wrt_h = [[angle_between(w_td[i,:], h_pcs[j,:]) for i in range(w_td.shape[0])] for j in range(h_pcs.shape[0])]

    # 5. Extract activity during the critical task periods and save
    timing = {'dead_time'  : 300,
              'fix_time'   : 200,
              'sample_time': 300,
              'delay_time' : 1000,
              'test_time'  : 300}
    h_by_time = {}
    start_time = 0
    for k, v in timing.items():
        h_by_time[k[:k.find("_")]] = h[:, start_time:start_time + v // params.dt]
        start_time += v // params.dt

    # 6. Also save out the labels of trials by task
    task_ids = trial_info[-1]
    sample = trial_info[-2][:,0]

    # Do the same thing, but to eliminate variability due to stimulus, stratify 
    # by stimulus ID
    fig, ax = plt.subplots(nrows=len(np.unique(sample)), ncols=len(timing.keys())-1, figsize=(10,13))
    for j, s in enumerate(np.unique(sample)):
        for i, (k, v) in enumerate(h_by_time.items()):
            if i < 1:
                continue
            rel_h = v[np.where(sample == s)[0],...]
            #print(len(np.where(sample == s)[0]))
            pcs = PCA(2).fit_transform(rel_h.reshape(-1, h.shape[-1]))
            rules = np.repeat(task_ids[np.where(sample == s)[0]], v.shape[1])
            ax[j,i-2].scatter(pcs[:,0], pcs[:,1], c=rules, cmap=plt.cm.rainbow)
            ax[j,i-2].set(title=f"{k}, sample={s}")
    plt.tight_layout()
    fig.savefig(fn, dpi=300)
    plt.close(fig)

    # 7. Identify the subset of output neurons that are driven by test-period activity
    # for each of the tasks in isolation
    output_modulation_scores_0 = np.zeros((h.shape[-1], len(np.unique(task_ids))))
    output_modulation_scores_1 = np.zeros((h.shape[-1], len(np.unique(task_ids))))
    output_modulation_scores_2 = np.zeros((h.shape[-1], len(np.unique(task_ids))))
    for i, t in enumerate(np.unique(task_ids)):
        rel_h = h_by_time['test'][np.where(task_ids == t)[0],...]
        output_proj = np.zeros((h.shape[-1], 2))
        for trial in range(rel_h.shape[0]):
            for timepoint in range(rel_h.shape[1]):
                output_proj[:,0] += np.multiply(rel_h[trial, timepoint,:], w_out[:,1])
                output_proj[:,1] += np.multiply(rel_h[trial, timepoint,:], w_out[:,2])
        output_modulation_scores_1[:,i] = output_proj[:,0] / (rel_h.shape[0] * rel_h.shape[1])
        output_modulation_scores_2[:,i] = output_proj[:,1] / (rel_h.shape[0] * rel_h.shape[1])

    for i, t in enumerate(np.unique(task_ids)):
        rel_h = h_by_time['fix'][np.where(task_ids == t)[0],...]
        output_proj = np.zeros((h.shape[-1], 1))
        for trial in range(rel_h.shape[0]):
            for timepoint in range(rel_h.shape[1]):
                output_proj[:,0] += np.multiply(rel_h[trial, timepoint,:], w_out[:,0])
        output_modulation_scores_0[:,i] = output_proj[:,0] / (rel_h.shape[0] * rel_h.shape[1])

    # Divide each row by its max value
    output_modulation_scores_0 /= np.maximum(np.amax(output_modulation_scores_0, axis=1)[:,np.newaxis], 1e-9)
    output_modulation_scores_1 /= np.maximum(np.amax(output_modulation_scores_1, axis=1)[:,np.newaxis], 1e-9)
    output_modulation_scores_2 /= np.maximum(np.amax(output_modulation_scores_2, axis=1)[:,np.newaxis], 1e-9)
    

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    # Row 0: sorted according to 0 order
    # Row 1: sorted according to 1 order
    # Row 2: sorted according to sum order
    all_mods = [output_modulation_scores_0, output_modulation_scores_1, output_modulation_scores_2]
    km_0 = KMeans(len(np.unique(task_ids))+1).fit(output_modulation_scores_0)
    km_1 = KMeans(len(np.unique(task_ids))+1).fit(output_modulation_scores_1)
    km_2 = KMeans(len(np.unique(task_ids))+1).fit(output_modulation_scores_2)
    all_km = [km_0, km_1, km_2]
    for i in range(3):
        for j in range(3):
            sorted_output_mod = all_mods[i][np.argsort(all_km[j].labels_),:]
            ax[i,j].imshow(sorted_output_mod, aspect='auto') # row = mods, col = sorted as
    #sorted_output_mod_s = output_modulation_sum[np.argsort(km_s.labels_),:]
    #ax.imshow(sorted_output_mod_s, aspect='auto')
    plt.tight_layout()
    
    fig.savefig("_".join([fn_beg, "output_clust", fn_ext]), dpi=300)
    plt.close(fig)

    print(0/0)
    # For each unit, print out the preferred task for each output unit (-1 if that unit not active)
    all_prefs = np.zeros((h.shape[-1], 3))
    for n in range(h.shape[-1]):
        for i in range(3):
            pref_i = all_mods[i][n,:]
            if np.amax(pref_i) < 1.0:
                all_prefs[n,i] = -1
            else:
                all_prefs[n,i] = np.argmax(pref_i)
    np.save("_".join([fn_beg, "task_output_prefs", ".npy"]), all_prefs)

    # To examine how task-specific processing is through time up until the test period (given that
    # we know the outputs are read out from separate sets of neurons), compute within-task vs.
    # across-task distance at each timepoint (marginalizing out sample ID)
    # Plot: ratio of within- to across-task distance at each timepoint
    test_start = (timing['dead_time'] + timing['fix_time'] + 
        timing['sample_time'] + timing['delay_time']) // params.dt
    within_task_distances = np.zeros((test_start, len(np.unique(task_ids)),len(np.unique(sample))))
    between_task_distances = np.zeros((test_start, len(np.unique(task_ids)),len(np.unique(sample))))
    
    for i,t in enumerate(np.unique(task_ids)):
        for j,s in enumerate(np.unique(sample)):
            for timepoint in range(test_start):
                cur_task_h = h[:,timepoint,:]
                cur_task_h = cur_task_h[np.where((sample == s) & (task_ids == t))[0],...]
                other_tasks_h = h[:,timepoint,:]
                other_tasks_h = other_tasks_h[np.where((sample == s) & (task_ids != t))[0],...]
                wtd = scipy.spatial.distance.pdist(cur_task_h)
                btd = scipy.spatial.distance.cdist(cur_task_h, other_tasks_h)
                within_task_distances[timepoint,i,j] = np.mean(wtd)
                between_task_distances[timepoint,i,j] = np.mean(btd)

    fig, ax = plt.subplots(1)
    dist_ratio = np.nanmean(within_task_distances,(1,2))/np.nanmean(between_task_distances,(1,2))
    ax.plot(dist_ratio)
    ax.set(ylabel="WTD / BTD", xlabel="Timestep", title="Ratio of within- to between-task distance vs. time")
    plt.tight_layout()
    fig.savefig("_".join([fn_beg, "wcd_bcd", fn_ext]), dpi=300)
    plt.close(fig)

    # # Compute ratio of activity in each cluster wrt output projection at each timepoint; 
    # # plot the mean of this ratio across tasks, as well as the ratio for each of the tasks separately
    # cluster_mappings = {}
    # for l in np.unique(km_s.labels_):
    #     rel_mod = output_modulation_sum[np.where(km_s.labels_ == l)[0],:]
    #     if np.amax(rel_mod) < 1.0:
    #         print(l)
    #         continue
    #     task_pref = np.argmax(np.mean(rel_mod,axis=0))
    #     print(l, task_pref)
    #     cluster_mappings[task_pref] = l
    # subpopulation_ratios = np.zeros((len(np.unique(task_ids)), h.shape[1]))
    # for i,t in enumerate(np.unique(task_ids)):
    #     rel_h = h[...,np.where(km.labels_ == cluster_mappings[t])]
    #     for timepoint in range(h.shape[1]):
    #         cur_task_h = rel_h[:,timepoint,:].squeeze()
    #         cur_task_h = cur_task_h[np.where(task_ids == t),...].squeeze()
    #         other_tasks_h = rel_h[:,timepoint,:].squeeze()
    #         other_tasks_h = other_tasks_h[np.where(task_ids != t),...].squeeze()


    #         subpopulation_ratios[i,timepoint] = np.mean(cur_task_h) - np.mean(other_tasks_h)

    # fig, ax = plt.subplots(1)
    # print(np.nanmean(subpopulation_ratios,0))
    # ax.plot(np.nanmean(subpopulation_ratios,0), color='black', linewidth=2)
    # ax.plot(subpopulation_ratios.T)
    # ax.set(xlabel='Timestep', ylabel='Subpopulation activity ratio')
    # plt.tight_layout()
    # fig.savefig("_".join([fn_beg, "subpop_ratio", fn_ext]), dpi=300)


    # - Use Murray-type approach to identify task-PCs (average across all sample stimuli, all timepoints,
    # all trials, stratified by task ID; then do PCA on this collection of N-dimensional vectors to find
    # the directions along which task context can be discriminated)
    task_by_neur = [np.mean(h[np.where(task_ids == t)[0],...], axis=(0,1)) for t in np.unique(task_ids)]
    task_pca = PCA(args.n_pcs).fit(task_by_neur)
    task_pcs = task_pca.components_
    var_exp = task_pca.explained_variance_ratio_

    # Make elbow plot to determine how many of these dimensions capture task-specific variance
    fig, ax = plt.subplots(1)
    ax.plot(var_exp)
    ax.set(xlabel='PC #', ylabel='Pct. variance explained')
    plt.tight_layout()
    fig.savefig("_".join([fn_beg, "elbow_plot", fn_ext]), dpi=300)
    plt.close(fig)

    # Separate approach: PCA on each task's activity separately, 
    # and in each epoch separately (to remove effect of sample);
    # then project activity onto these PCs for this task vs. other
    # tasks to determine whether activity related to other tasks
    # enters/encroaches on the subspace; then project other tasks
    # into this task's subspace and then onto output, and compare with
    # this task's projection onto output (prediction: will be very close to 
    # matching in magnitude across each of the decisions, making this task's
    # activity in the perfect subspace to tilt the competition)
    projections = np.zeros((len(np.unique(task_ids)),len(np.unique(task_ids)), len(h_by_time.keys())))
    output_projections = np.zeros((len(np.unique(task_ids)),len(np.unique(task_ids)), len(h_by_time.keys()),3))
    output_projections_raw = np.zeros((len(np.unique(task_ids)),len(np.unique(task_ids)), len(h_by_time.keys()),3))
    pcas = {}
    for i, t1 in enumerate(np.unique(task_ids)):
        for j, t2 in enumerate(np.unique(task_ids)):
            for k, (key, v) in enumerate(h_by_time.items()):

                fit_h = v[np.where(task_ids == t1),...]
                trans_h = v[np.where(task_ids == t2),...]
                fit_h_to_fit = np.reshape(fit_h, (-1, fit_h.shape[-1]))
                trans_h_to_trans = np.reshape(trans_h, (-1, trans_h.shape[-1]))
                task_pca = PCA(args.n_pcs).fit(fit_h_to_fit)
                trans_proj = task_pca.transform(trans_h_to_trans) # t2 trials x N_PCs
                pcas[(t1, key)] = task_pca
                projections[i,j,k] = np.mean(trans_proj)

                output_adjusted = task_pca.transform(w_out.T).T # yields N_PCs x 3
                output_projections[i,j,k,:] = np.mean(trans_proj @ output_adjusted, axis=0)
                #output_projections_raw[i,j,k,:]

    np.save("_".join([fn_beg, "projections", ".npy"]), projections)
    np.save("_".join([fn_beg, "output_projections", ".npy"]), output_projections)
    #np.save("_".join([fn_beg, "output_projections_raw", ".npy"]), output_projections_raw)
    print(0/0)


    # - Identify each task's separate output axis (projection ofo )
    # - Measure angle between 

    # Measure angle between 

    return np.array(angles_wrt_output), np.array(angles_wrt_h), task_ids

if __name__ == "__main__":
    fns = get_fns()
    if not os.path.exists(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    angles_wrt_outputs = []
    angles_wrt_h = []
    #hs_by_time = []
    task_ids = []
    accs = []
    for fn in fns:
        data = load_data(fn)
        print(os.path.split(fn)[1])
        if data[4] < 0.7:
            continue
        save_fn = os.path.join(args.figure_save_path, os.path.split(fn[:-4])[1] + ".png")
        angles_o, angles_h, task_id = analyze_output_proximity(*data, save_fn)
        angles_wrt_outputs.append(angles_o)
        angles_wrt_h.append(angles_h)
        #hs_by_time.append(h_by_time)
        task_ids.append(task_id)
        accs.append(data[4])

        if "fold=7_number=1_acc=0.8966" in fn:
            print(0/0)

    angles_wrt_outputs = np.array(angles_wrt_outputs)
    angles_wrt_h = np.array(angles_wrt_h)
    #hs_by_time = np.array(hs_by_time)
    task_ids = np.array(task_ids)
    accs = np.array(accs)
    np.save(os.path.join(args.figure_save_path, "angles_wrt_outputs.npy"), angles_wrt_outputs)
    np.save(os.path.join(args.figure_save_path, "angles_wrt_h.npy"), angles_wrt_h)
    #np.save(os.path.join(args.figure_save_path, "hs_by_time.npy"), hs_by_time)
    np.save(os.path.join(args.figure_save_path, "task_ids.npy"), task_ids)
    np.save(os.path.join(args.figure_save_path, "accs.npy"), accs)

    