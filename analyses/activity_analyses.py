import pickle, glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding, Isomap
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy import stats, spatial
from scipy.spatial.distance import pdist, cdist
from collections import defaultdict
import warnings, argparse
import os, time, datetime
import scipy.stats, scipy.special
import scipy.ndimage as sim
import h5py, pandas as pd, seaborn as sns
import itertools
from matplotlib.gridspec import GridSpec
from src.util import str2bool

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'CMU Sans Serif'
np.set_printoptions(precision=3, suppress=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser('')
parser.add_argument('--data_path', type=str, default='/media/graphnets/reservoir/experiment2/')
parser.add_argument('--figure_save_path', type=str, default='/media/graphnets/reservoir/figures/')
parser.add_argument('--analysis_save_path', type=str, default='/media/graphnets/reservoir/analysis/')
parser.add_argument('--N_PCS', type=int, default=5)
parser.add_argument('--N_MAX_PCS', type=int, default=5)
parser.add_argument('--k_folds', type=int, default=4)
parser.add_argument('--do_plot', type=str2bool, default=False)
parser.add_argument('--do_save', type=str2bool, default=True)
parser.add_argument('--do_overwrite', type=str2bool, default=False)
parser.add_argument('--n_perturbation_fracs', type=int, default=1)
parser.add_argument('--n_repetitions', type=int, default=1)
parser.add_argument('--do_learning_analysis', type=str2bool, default=True)
parser.add_argument('--n_estimators', type=int, default=10)
parser.add_argument('--dt', type=int, default=20)

warnings.filterwarnings('ignore')
args = parser.parse_args()


np.set_printoptions(precision=3)

class NetworkAnalyzer:

    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

        # Split up filename
        self.fn_beg, self.fn_ext = os.path.splitext(fn)
        self.network_id = os.path.split(self.fn_beg)[1]

        # Prepare path for saving figures/analysis
        self.fig_path = os.path.join(self.args.figure_save_path, 
            self.network_id)
        self.analysis_path = os.path.join(self.args.analysis_save_path,
            self.network_id)

        # Load and bind data for this network
        data = self.load_data(self.fn)
        if not type(data) is tuple:
            self.analysis_save_f = data
            return 
        self.h, self.w_td, self.w_rec, self.w_out, self.acc, \
            self.task_ids, self.sample, self.labels = data

        # Extract some key elements/shapes
        self.N_TASKS   = len(np.unique(self.task_ids))
        self.N_SAMPLES = len(np.unique(self.sample))
        self.N_NEUR    = self.h.shape[-1]
        self.N_DECISIONS = len(np.unique(self.labels))
        self.T         = self.h.shape[1]

        self.task_list = np.array(["LeftIndication", "RightIndication", 
            "MinIndication", "MaxIndication", "SubtractingLeftRight",
            "SubtractingRightLeft", "Summing", "SummingOpposite"])
        self.output_list = np.array([f'{i}' + u'\N{DEGREE SIGN}' \
            for i in [0, 45, 90, 135, 180, 225, 270, 315]])

        # Prepare to extract activity during the unique task epochs
        timing = {'dead_time'  : 300,
                  'fix_time'   : 200,
                  'sample_time': 300,
                  'delay_time' : 1000,
                  'test_time'  : 300}

        self.epoch_bounds = {}
        start_time = 0
        for k, v in timing.items():
            cur_rng = range(start_time, start_time + v // self.args.dt)
            self.epoch_bounds[k[:k.find("_")]] = cur_rng
            start_time += v // self.args.dt


    def load_data(self, fn):

        analysis_save_f = f"{self.analysis_path}_analysis.pkl"

        if not self.args.do_overwrite and os.path.exists(analysis_save_f):
            return analysis_save_f

        # Load and attach data
        data = pickle.load(open(fn, 'rb'))
        alt_data = pickle.load(open(fn[:-7] + ".pkl", 'rb'))
        if len(data.keys()) == 0:
            return 0
        
        w_td       = data['top_down0_w:0'][-1] @ data['top_down1_w:0'][-1]
        w_rec      = data['rnn_w:0'][-1]
        w_out      = data['policy_w:0'][-1]
        self.b_out = data['policy_b:0'][-1].astype(np.float32)
        self.w_in  = data['bottom_up_w:0'][-1][:-3,:].astype(np.float32)

        acc = np.mean(np.array(alt_data['task_accuracy'])[:, -25:, :], axis=(0,1))

        # Preserve activity from training (assume this is present for all nets;
        # won't analyze the nets that were mediocre performers)
        self.train_acts    = h5py.File(fn[:-7] + ".h5",'r')
        t0 = time.time()
        h                  = self.train_acts['data'][-1,:,:,:].astype(np.float32)
        self.train_rules   = data['train_rules'][0].astype(np.int32)
        self.train_accs    = data['train_accs'][0]
        self.train_labels  = data['train_labels'][0].astype(np.int32)
        self.train_samples = data['train_samples'][0].astype(np.int32)
        self.train_w_outs  = data['policy_w:0']
        self.train_w_ins   = data['bottom_up_w:0']
        self.train_w_td_0s = data['top_down0_w:0']
        self.train_w_td_1s = data['top_down1_w:0']
        self.train_b_outs  = data['policy_b:0']
        self.train_iters   = data['save_iters']

        task_ids = self.train_rules
        sample   = self.train_samples
        labels   = self.train_labels

        # Assemble counterfactual output array (N_TRIALS x N_TASKS)
        self.ctfctl_y = np.zeros((task_ids.shape[0], len(np.unique(task_ids))))
        for tri in range(self.ctfctl_y.shape[0]):
            s = sample[tri]
            self.ctfctl_y[tri,0] = s[0] + 1
            self.ctfctl_y[tri,1] = s[1] + 1
            self.ctfctl_y[tri,2] = np.amin(s) + 1
            self.ctfctl_y[tri,3] = np.amax(s) + 1
            self.ctfctl_y[tri,4] = (s[0] - s[1]) % 8 + 1
            self.ctfctl_y[tri,5] = (s[1] - s[0]) % 8 + 1
            self.ctfctl_y[tri,6] = (s[0] + s[1]) % 8 + 1
            self.ctfctl_y[tri,7] = (s[0] + s[1] + 4) % 8 + 1


        return h, w_td, w_rec, w_out, acc, task_ids, sample, labels

    def get_colors(self):
        colors = plt.cm.jet(np.linspace(0,1,self.N_TASKS))
        lightdark_colors = plt.cm.tab20(np.linspace(0, 1, self.N_TASKS*2))
        light_colors = lightdark_colors[::2]
        dark_colors = lightdark_colors[1::2]

        return colors, light_colors, dark_colors

    def perform_all_analyses(self, to_do=np.arange(20, 40), DO_REDUCE_H=False):

        results = {}

        # If file already exists, don't overwrite unless specified
        analysis_save_f = f"{self.analysis_path}_analysis.pkl"
        if os.path.exists(analysis_save_f):
            results = pickle.load(open(analysis_save_f, 'rb'))
            print("Loaded.")
        if not self.args.do_overwrite and len(results.keys()) > 0:
            print(analysis_save_f)
            return results, analysis_save_f

        t0 = time.time()

        ########################################################################
        # Identifying context-dependent and -independent dimensions of activity
        ########################################################################
        # 2. Identify context-PCs within full network activity space
        if 2 in to_do:
            context_pcs, ctx_pcs_exp_var = self.identify_context_subspace(self.h)
            results['context_pcs_exp_var'] = ctx_pcs_exp_var

        ########################################################################
        # Decoding context from activity
        ########################################################################

        # 13. Cross-temporal decode of context, raw
        if 13 in to_do:
            cross_temp_ctx_raw = self.cross_temporal_decode(self.h,
                self.task_ids, save_fn="cross_temp_context_decode_raw", do_reduce=True)
            results['cross_temp_ctx_raw'] = cross_temp_ctx_raw
         
        # 14. Cross-temporal decode of context, context subspace
        if 14 in to_do:
            cross_temp_ctx_td = self.cross_temporal_decode(self.h, self.task_ids,
                context_pcs, 
                save_fn="cross_temp_context_decode_task_dependent_subspace")
            results['cross_temp_ctx_td'] = cross_temp_ctx_td

        # 16. Cross-temporal decode of samples, raw
        if 16 in to_do:
            cross_temp_sam_raw = self.cross_temporal_decode(self.h,
                self.sample, decode_key='sample', 
                save_fn="cross_temp_sample_decode_raw", do_reduce=True)
            results['cross_temp_sam_raw'] = cross_temp_sam_raw

        ########################################################################
        # Learning-related analyses
        ########################################################################
        var_alignment          = []
        var_alignment_mis      = []
        rand_var_alignment     = []
        var_alignment_by_epoch = []
        var_alignment_mis_by_epoch = []
        rand_var_alignment_by_epoch = []
        var_alignment_by_epoch_no_stim = []
        var_alignment_mis_by_epoch_no_stim = []
        rand_var_alignment_by_epoch_no_stim = []
        counterfactual_decode  = []
        task_dimensionality    = []
        exp_var_pcts           = []
        sam_per_ctx_dec        = []
        out_per_ctx_dec        = []
        pairwise_task_angles_reduced = []
        pairwise_task_angles_full    = []
        pairwise_task_angles_n_pcs   = []
        rd_decodes = []
        rd_decodes_by_time = []
        s0_decodes = []
        s1_decodes = []
        o_decodes  = []
        sc_corrs = []
        subpop_scores = []
        task_agnosticity = []
        shared_output_subspaces = []
        ensemble_recruitment_similarities = []

        t0_tr = time.time()
        for i, iteration in enumerate(self.train_iters):
            print(f"\t{i}")


            h_i = self.train_acts['data'][i,:,:,:].astype(np.float32)

            # Remove components directly related to stimulus presentation
            '''
            to_remove = []
            for samp in np.unique(self.train_samples, axis=0):
                samp_trials = np.unique(np.where(self.train_samples == samp)[0])
                act = h_i[samp_trials]
                to_remove.append(act[:, self.epoch_bounds['sample'],:].mean((0)))
            '
            
            sample_period_pcs = PCA().fit(np.array(to_remove).reshape((-1, self.N_NEUR)))
            n_pcs_to_remove = np.argwhere(np.cumsum(sample_period_pcs.explained_variance_ratio_) > 0.99)[0][0]
            h_to_change = np.reshape(h_i, (-1, self.N_NEUR))
            h_to_change = np.dot(h_to_change - sample_period_pcs.mean_, sample_period_pcs.components_.T[:,n_pcs_to_remove:])
            h_to_change = np.dot(h_to_change, sample_period_pcs.components_[n_pcs_to_remove:, :]) + sample_period_pcs.mean_
            h_i = np.reshape(h_to_change, h_i.shape)

            
            self.obtain_output_subspace(h_i, self.train_rules, self.train_labels)
            '''

            if DO_REDUCE_H:
                # Reduce in dimensionality to N PCs that explain >99 percent of variance
                full_pca = PCA().fit(np.reshape(h_i, (-1, self.N_NEUR)))
                e_v = np.cumsum(full_pca.explained_variance_ratio_)
                n_pcs_full = self.N_NEUR - len(e_v[e_v > 0.99])
                h_i = np.reshape(
                    PCA(n_pcs_full).fit_transform(
                        np.reshape(h_i, (-1, self.N_NEUR))),
                        (h_i.shape[0], h_i.shape[1], n_pcs_full))

            # 22. Compute variance alignment analysis
            if 22 in to_do:
                v_a_i, v_a_i_m, rv_a_i = self.compute_variance_alignment(h_i, self.train_rules)
                var_alignment.append(v_a_i)
                var_alignment_mis.append(v_a_i_m)
                rand_var_alignment.append(rv_a_i)
                print(f"\t\t 22: {str(datetime.timedelta(seconds=int(round(time.time() - t0_tr, 0))))}")

            # 23. Epoch-dependent variance alignment analysis
            if 23 in to_do:
                v_a_i, v_a_i_m, rv_a_i = self.compute_variance_alignment_by_epoch(h_i, 
                        self.train_rules)
                var_alignment_by_epoch.append(v_a_i)
                var_alignment_mis_by_epoch.append(v_a_i_m)
                rand_var_alignment_by_epoch.append(rv_a_i)
                print(f"\t\t 23: {str(datetime.timedelta(seconds=int(round(time.time() - t0_tr, 0))))}")

            # 23a. Epoch-dependent variance alignment analysis (axis of sample stimulus driving subtracted)
            if "23a" in to_do:
                v_a_i, v_a_i_m, rv_a_i = self.compute_variance_alignment_by_epoch_no_stim_comp(h_i, 
                        self.train_rules, self.train_samples, self.train_w_ins[i])
                var_alignment_by_epoch_no_stim.append(v_a_i)
                var_alignment_mis_by_epoch_no_stim.append(v_a_i_m)
                rand_var_alignment_by_epoch_no_stim.append(rv_a_i)
                print(f"\t\t 23a: {str(datetime.timedelta(seconds=int(round(time.time() - t0_tr, 0))))}")

            # 24. Compute dimensionality of each task's activity in each epoch
            if 24 in to_do:
                t_d, e_v = self.compute_dimensionality_by_task(h_i, self.train_rules)
                task_dimensionality.append(t_d)
                exp_var_pcts.append(e_v)
                print(f"\t\t 24: {str(datetime.timedelta(seconds=int(round(time.time() - t0_tr, 0))))}")

            # 25. Counterfactual decoding analysis
            if 25 in to_do:
                ctf_d_i, ctf_ov = self.decode_counterfactual(h_i, self.train_rules,
                    self.train_samples, self.train_labels)
                counterfactual_decode.append(ctf_d_i)
                if i == 0:
                    counterfactual_overlap = ctf_ov
                print(f"\t\t 25: {str(datetime.timedelta(seconds=int(round(time.time() - t0_tr, 0))))}")

            # 28. Pairwise task angles
            if 28 in to_do:
                w_td = self.train_w_td_0s[i] @ self.train_w_td_1s[i]
                p_t_a_red, p_t_a_full, n_pcs = \
                    self.compute_pairwise_task_angles(w_td)
                pairwise_task_angles_reduced.append(p_t_a_red)
                pairwise_task_angles_full.append(p_t_a_full)
                pairwise_task_angles_n_pcs.append(n_pcs)
                print(f"\t\t 28: {str(datetime.timedelta(seconds=int(round(time.time() - t0_tr, 0))))}")

            # 29. Readout dimensionality
            if "29a" in to_do:
                rd_dec, rd_dec_by_t = self.compute_readout_dimensionality(h_i, self.train_rules,
                    self.train_labels, self.train_w_outs[i], self.train_b_outs[i])
                rd_decodes.append(rd_dec)
                rd_decodes_by_time.append(rd_dec_by_t)
                print(f"\t\t 29: {str(datetime.timedelta(seconds=int(round(time.time() - t0_tr, 0))))}")

            # 30. Subspace specialization
            if 30 in to_do:
                s0_dec, s1_dec, o_dec = self.compute_subspace_specialization(h_i, self.train_rules, self.train_labels,
                    self.train_samples)
                s0_decodes.append(s0_dec)
                s1_decodes.append(s1_dec)
                o_decodes.append(o_dec)
                print(f"\t\t 30: {str(datetime.timedelta(seconds=int(round(time.time() - t0_tr, 0))))}")

            # 31. Output subpopulation test (task agnosticity index)
            if 31 in to_do:
                sc_corr, subpop_sc, tai = self.compute_output_separation_subpopulations(h_i, 
                    self.train_rules, self.train_labels, self.train_w_outs[i])
                sc_corrs.append(sc_corr)
                subpop_scores.append(subpop_sc)
                task_agnosticity.append(tai)
                print(f"\t\t 31: {str(datetime.timedelta(seconds=int(round(time.time() - t0_tr, 0))))}")

            # 32. Ensemble overlap analysis
            if 32 in to_do:
                w_td = self.train_w_td_0s[i] @ self.train_w_td_1s[i]
                e_r_s = self.compute_ensemble_recruitment_similarity(w_td)
                ensemble_recruitment_similarities.append(e_r_s)
                print(f"\t\t 32: {str(datetime.timedelta(seconds=int(round(time.time() - t0_tr, 0))))}")

            # 100. Decode sample stimulus in each task's specific subspace
            # (bonus: in each epoch!)
            if 100 in to_do:
                s_p_c_d = self.decode_by_task_and_epoch(h_i,
                    self.train_rules, self.train_samples)
                sam_per_ctx_dec.append(s_p_c_d)

            # 101. Decode output in each task's specific subspace
            # (bonus: in each epoch!)
            if 101 in to_do:
                o_p_c_d = self.decode_by_task_and_epoch(h_i,
                    self.train_rules, self.train_labels)
                out_per_ctx_dec.append(o_p_c_d)


        if len(var_alignment) > 0:
            results['var_alignment'] = var_alignment
            results['var_alignment_mis'] = var_alignment_mis
            results['rand_var_alignment'] = rand_var_alignment
        if len(var_alignment_by_epoch) > 0:
            results['var_alignment_by_epoch'] = var_alignment_by_epoch
            results['var_alignment_mis_by_epoch'] = var_alignment_mis_by_epoch
            results['rand_var_alignment_by_epoch'] = rand_var_alignment_by_epoch
        if len(var_alignment_by_epoch_no_stim) > 0:
            results['var_alignment_by_epoch_no_stim'] = var_alignment_by_epoch_no_stim
            results['var_alignment_mis_by_epoch_no_stim'] = var_alignment_mis_by_epoch_no_stim
            results['rand_var_alignment_by_epoch_no_stim'] = rand_var_alignment_by_epoch_no_stim
        if len(counterfactual_decode) > 0:
            results['counterfactual_decode'] = np.array(counterfactual_decode)
            results['counterfactual_overlap'] = counterfactual_overlap
        if len(task_dimensionality) > 0:
            results['task_dimensionality'] = np.array(task_dimensionality)
            results['exp_var_pcts'] = np.array(exp_var_pcts)
        if len(sam_per_ctx_dec) > 0:
            results['sam_per_ctx_dec'] = np.array(sam_per_ctx_dec)
        if len(out_per_ctx_dec) > 0:
            results['out_per_ctx_dec'] = np.array(out_per_ctx_dec)
        if len(pairwise_task_angles_reduced) > 0:
            results['pairwise_task_angles_reduced'] = pairwise_task_angles_reduced
            results['pairwise_task_angles_full'] = pairwise_task_angles_full
            results['pairwise_task_angles_n_pcs'] = pairwise_task_angles_n_pcs
        if len(s0_decodes) > 0:
            results['s0_decodes'] = np.array(s0_decodes)
            results['s1_decodes'] = np.array(s1_decodes)
            results['o_decodes']  = np.array(o_decodes)
        if len(rd_decodes) > 0:
            results['rd_decodes'] = np.array(rd_decodes)
        if len(subpop_scores) > 0:
            results['sc_corrs'] = np.array(sc_corrs)
            results['subpop_scores'] = np.array(subpop_scores)
            results['task_agnosticity'] = np.array(task_agnosticity)
        if len(ensemble_recruitment_similarities) > 0:
            results['ensemble_recruitment_similarities'] = np.array(ensemble_recruitment_similarities)

        results['accs'] = self.train_accs

        if self.args.do_save:
            pickle.dump(results, open(analysis_save_f, 'wb'))

        # Print the network name
        t = str(datetime.timedelta(seconds=int(round(time.time() - t0, 0))))

        print(self.network_id, t, f"{self.acc.mean():.2f}")

        return results, analysis_save_f


    def identify_context_subspace(self, h, do_plot=False):
        """
        Perform Murray-type analysis to identify subspace for consistent
        discrimination of task ID. (PCA with 7 input samples, one per task,
        consisting of average activity vector across all trials/timepoints
        during that task).

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
        Returns:
            context_pca (sklearn PCA() object) - basis of context subspace
            var_exp (np.ndarray) - cumulative variance explained for comp. 1-N
        """
        # Average activity across all timepoints/trials within each task
        task_tr = [np.where(self.task_ids == i)[0] for i in 
            np.unique(self.task_ids)]
        mean_task_h = np.array([h[t,...].mean((0,1)) for t in task_tr])

        # Perform PCA on tasks x N matrix to identify task subspace
        context_pca = PCA(self.args.N_PCS).fit(mean_task_h)
        context_pcs = context_pca.components_

        # Project all activity into this subspace + compute variance
        # explained per component
        all_data = np.reshape(h, (-1, self.N_NEUR))
        var_exp  = np.zeros(self.args.N_PCS)
        for i in range(self.args.N_PCS):
            pca_i = PCA(i + 1).fit(mean_task_h)
            var_exp[i] = r2_score(all_data, 
                                  pca_i.inverse_transform(
                                    pca_i.transform(all_data)),
                                  multioutput='variance_weighted')      
        var_exp = np.array(var_exp)

        # If plotting: 
        # (a) project activity from all trials/timepoints into the first 2 PCs 
        #     of this subspace, color by task, and plot
        # (b) project time-averaged activity in each task epoch from every trial
        #     into the first 2 PCs of this subspace, color by task, and plot
        #     each in a separate subplot
        if self.args.do_plot and do_plot:
            colors, _, _ = self.get_colors()

            fig = plt.figure(constrained_layout=True, figsize=(10,5))
            gs = fig.add_gridspec(2,5)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[1, 2])
            ax5 = fig.add_subplot(gs[1, 3])
            ax6 = fig.add_subplot(gs[1, 4])
            ax = [ax1, ax2, ax3, ax4, ax5, ax6]

            for i, t in enumerate(np.unique(self.task_ids)):

                # (a) Plot of all timepoints/trials
                h_orig = h[task_tr[i],...].reshape((-1, self.N_NEUR))
                h_trans = context_pca.transform(h_orig)[:,:2]
                h_trans = np.reshape(h_trans, 
                    (h[task_tr[i]].shape[0], h[task_tr[i]].shape[1], 2))
                for trial in h_trans:
                    ax[0].plot(trial[:,0], trial[:,1], color=colors[i])

                # (b) Plot of mean across timepoints per epoch, for all trials
                for j, (k,b) in enumerate(self.epoch_bounds.items()):
                    h_orig = h[task_tr[i],...]
                    h_orig = h_orig[:,b,:].mean(1)
                    h_trans = context_pca.transform(h_orig)[:,:2]
                    ax[j + 1].scatter(h_trans[:,0], h_trans[:,1], color=colors[i])

            # Add titles/labels etc
            ax[0].set(title="All trials/timepoints", xlabel="PC1", ylabel="PC2")
            for j, (k, v) in enumerate(self.epoch_bounds.items()):
                ax[j + 1].set(title=k, xlabel="PC1", ylabel="PC2")
            fig_fn = f"{self.fig_path}_task_subspace_projections.png"
            fig.savefig(fig_fn, dpi=300)
            plt.close(fig)

        return context_pca, var_exp


    def decode(self, h, y, context_comp=None, 
        save_fn=None, do_plot=True, raw_n_comp=50, do_reduce=False,
        return_all=False, decode_key='context'):
        """
        Decode label (task ID, sample, etc.) from activity.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            y (np.ndarray) - labels (B x 1)
            context_comp (sklearn PCA() obj.) - context subsp. to project into
        Returns:
            scores (np.ndarray, (T,)) - decoding of context @ each timepoint
        """
        # If there are multiple outputs to decode: handle each separately
        if len(y.shape) < 2:
            y = y[...,np.newaxis]

        # Generate k-folds
        skf = StratifiedKFold(n_splits=self.args.k_folds)
        scores = np.zeros((y.shape[-1], h.shape[1], self.args.k_folds))
        svm = SVC(C=1.0, kernel='linear', max_iter=1000, 
            decision_function_shape="ovr", shrinking=False, tol=1e-3)

        # If any components specified, project onto them before decode
        if context_comp is not None:
            h_to_project = np.reshape(h, (-1, h.shape[-1]))
            h_projected = context_comp.transform(h_to_project)
            h = np.reshape(h_projected, (h.shape[0], h.shape[1], -1))

        # If not specified, reduce in dimensionality to e.g. 200D
        elif do_reduce:
            h_to_project = np.reshape(h, (-1, h.shape[-1]))
            h_projected = PCA(raw_n_comp).fit_transform(h_to_project)
            h = np.reshape(h_projected, (h.shape[0], h.shape[1], -1))

        # For each output to decode:
        for j in range(y.shape[-1]):
            # For each fold: at each timepoint, decode and store
            for i, (tr_idx, te_idx) in enumerate(skf.split(h[:, 0, :], y[:,j])):
                y_tr, y_te = y[tr_idx,j], y[te_idx,j]
                for t in range(h.shape[1]):
                    X_tr, X_te = h[tr_idx, t, :], h[te_idx, t,  :]
                    ss = StandardScaler()
                    ss.fit(X_tr)
                    X_tr = ss.transform(X_tr)
                    X_te = ss.transform(X_te)
                    svm.fit(X_tr, y_tr)
                    scores[j, t, i] = svm.score(X_te, y_te)
        
        # If plotting: plot the decode through time (mean across folds)
        if self.args.do_plot and do_plot:

            colors, _, _ = self.get_colors()

            fig, ax = plt.subplots(1)
            ax.plot(scores.mean(1), label='Decode')
            ax.hlines(y=1./len(np.unique(y)), xmin=0, xmax=scores.shape[0] - 1, 
                color='black', linestyle='--', label='Chance')
            ax.set(title=f"Decode {decode_key}", xlabel="Timestep",
                ylabel="Accuracy",ylim=[0.,1.])
            ax.legend()
            plt.tight_layout()

            # Build filename 
            if save_fn is None:
                fig_fn = f"{self.fig_path}_context_decode.png"
            else:
                fig_fn = f"{self.fig_path}_{save_fn}.png"
            fig.savefig(fig_fn, dpi=300)
            plt.close(fig)

        if return_all:
            return scores

        return scores.mean(-1)

    def cross_temporal_decode(self, h, y, context_comp=None, 
        save_fn=None, do_plot=True, raw_n_comp=50, do_reduce=False,
        decode_key='context'):
        """
        Decode informaton (e.g. task, sample) from activity, training
        decoder at each timepoint t1 and testing at each timepoint t2
        to assess the stability of information encoding through time.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            y (np.ndarray) - labels (B x 1)
            context_comp (sklearn PCA() obj.) - context subsp. to project into
        Returns:
            scores (np.ndarray, (T,T)) - decoding of context @ each (train, test)
                timepoint
        """
        # If there are multiple outputs to decode: handle each separately
        if len(y.shape) < 2:
            y = y[...,np.newaxis]

        # Generate k-folds
        skf    = StratifiedKFold(n_splits=self.args.k_folds)
        scores = np.zeros((y.shape[-1], int(np.ceil(h.shape[1]/2)), 
            int(np.ceil(h.shape[1]/2)), self.args.k_folds))
        svm = SVC(C=1.0, kernel='linear', max_iter=1000, 
                decision_function_shape='ovr', shrinking=False, tol=1e-3)

        # If any components specified, project onto them before decode
        if context_comp is not None:
            h_to_project = np.reshape(h, (-1, h.shape[-1]))
            h_projected = context_comp.transform(h_to_project)
            h = np.reshape(h_projected, (h.shape[0], h.shape[1], -1))

        # If not specified, reduce in dimensionality to e.g. 200D
        elif do_reduce:
            h_to_project = np.reshape(h, (-1, h.shape[-1]))
            h_projected = PCA(raw_n_comp).fit_transform(h_to_project)
            h = np.reshape(h_projected, (h.shape[0], h.shape[1], -1))

        # For each fold: at each timepoint, train decoder;
        # test that decoder at every timepoint; and store
        for j in range(y.shape[-1]):
            for i, (tr_idx, te_idx) in enumerate(skf.split(h[:, 0, :], y[:,j])):
                y_tr, y_te = y[tr_idx,j], y[te_idx,j]
                for t_tr in range(0,h.shape[1],2):
                    for t_te in range(t_tr, h.shape[1],2):
                        X_tr = h[tr_idx, t_tr, :]
                        X_te = h[te_idx, t_te,  :]
                        ss = StandardScaler()
                        ss.fit(X_tr)
                        X_tr = ss.transform(X_tr)
                        X_te = ss.transform(X_te)
                        svm.fit(X_tr, y_tr)
                        scores[j, t_tr//2, t_te//2, i] = svm.score(X_te, y_te)
        
        # If plotting: plot the decode through time (mean across folds)
        if self.args.do_plot and do_plot:

            fig, ax = plt.subplots(1)
            im = ax.imshow(scores.mean(-1).squeeze(), 
                vmin=1.0/len(np.unique(y)),
                vmax=1.0)
            ax.set(title=f"Decode {decode_key}", xlabel="Timestep",
                ylabel="Accuracy")
            fig.colorbar(im)
            plt.tight_layout()

            # Build filename 
            if save_fn is None:
                fig_fn = f"{self.fig_path}_context_decode.png"
            else:
                fig_fn = f"{self.fig_path}_{save_fn}.png"
            fig.savefig(fig_fn, dpi=300)
            plt.close(fig)

        return scores.mean(-1)

    def compute_cumulative_dimensionality(self, h, rules):
        """
        Quantify cumulative dimensionality of all provided activity, using 
        method laid out by Cueva et al (2020 PNAS)

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
        Returns:
            cum_dim (np.ndarray) - cumulative dimensionality of neural traj.,
                (T x 1)
        """


        return 

    def compute_dimensionality_by_task(self, h, rules, do_plot=True, 
        save_fn=None):
        """
        Quantify dimensionality in each task-specific subspace, 
        specifically to assess the idea that trajectories are 
        chaotic/high-dimensional.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            outputs (np.ndarray) - req. decision identifier for batch (B x 1)
        Returns:
            dim_part_rat (np.ndarray) -- dimensionality as measured through
                participation ratio (N_TASKS x N_EPOCHS)
            var_exp (np.ndarray) -- cumulative variance explained by PCs 1:k
        """
        # If only subset of data provided, don't try to execute for all epochs
        
        if h.shape[1] < self.epoch_bounds['test'][-1]:
            # Participation ratio: fit PCA, take percent explained
            dim_part_rat = np.zeros(self.N_TASKS)
            exp_var = np.zeros((self.N_TASKS, self.N_NEUR))
            
            for i, t in enumerate(np.unique(rules)):
                h_t = h[np.where(rules == t)[0]]
                pca = PCA().fit(np.reshape(h_t, (-1, self.N_NEUR)))
                e_vals = pca.explained_variance_ratio_
                dim_part_rat[i] = np.sum(e_vals)**2 / np.sum(e_vals**2)
                exp_var[i] = np.cumsum(e_vals)

        else:
            # Participation ratio: fit PCA, take percent explained
            dim_part_rat = np.zeros((self.N_TASKS, len(self.epoch_bounds)))
            exp_var = np.zeros((self.N_TASKS, len(self.epoch_bounds), 
                self.N_NEUR))
            
            for i, t in enumerate(np.unique(rules)):
                for j, b in enumerate(self.epoch_bounds.values()):
                    h_t = h[np.where(rules == t)[0]]
                    pca = PCA().fit(np.reshape(h_t[:,b[-5:],:], (-1, self.N_NEUR)))
                    e_vals = pca.explained_variance_ratio_
                    dim_part_rat[i,j] = np.sum(e_vals)**2 / np.sum(e_vals**2)
                    exp_var[i, j] = np.cumsum(e_vals)

        return dim_part_rat, exp_var

    def compute_variance_alignment(self, h, rules, total_n_pcs=None, n_boot=1000):
        """
        Variance alignment analysis: 
        Project activity from each task onto subspace defined from activity
        during other tasks; compute variance explained (amount of variance in
        projection onto those axes vs. amount of variance when projected onto 
        own axes).

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
        Returns:
            var_ex (np.ndarray, (N_TASKS x 2)) -- variance 
                explained for each (activity, PC) pairing
        """
        # To start: perform PCA on all activity together
        h[np.where(~np.isfinite(h))] = 0.
        pca = PCA().fit(np.reshape(h, (-1, self.N_NEUR)))
        e_vals = pca.explained_variance_ratio_
        total_n_pcs = max(np.argwhere(np.cumsum(e_vals) > 0.9)[0][0] + 1, 2)
        print(total_n_pcs)
        #return None, None, None

        # Transform data
        h_to_trans = np.reshape(h, (-1, self.N_NEUR))
        comp = pca.components_
        mu = pca.mean_
        h_low_d = np.dot(h_to_trans - mu, comp.T[:,:total_n_pcs])
        h_low_d = np.reshape(h_low_d, (h.shape[0], h.shape[1], total_n_pcs))

        # Compute variance explainable of activity during each task j 
        # in first k components defined from task i activity
        # var_ex: first dimension = fit, second dimension = act., third = pcs
        var_ex = np.zeros((self.N_TASKS, self.N_TASKS, total_n_pcs))
        var_ex_mis = np.zeros_like(var_ex)

        # Generate control var_ex, too: 
        rand_var_ex = np.zeros((self.N_TASKS, total_n_pcs, n_boot))

        h_by_task = {j: h_low_d[np.where(rules == j)[0]] 
            for j in np.unique(rules)}
        h_by_task = {k: np.reshape(v, (-1, total_n_pcs)) 
            for k, v in h_by_task.items()}
        rule_options = np.unique(rules)
        
        per_task_vars = np.zeros((self.N_TASKS, total_n_pcs))

        for i, t1 in enumerate(rule_options):

            pca_i = PCA().fit(h_by_task[t1])
            comp  = pca_i.components_
            per_task_vars[i,:] = np.cumsum(
                np.var(np.dot(h_by_task[t1], comp.T), axis=0))
            for j, t2 in enumerate(rule_options):
                var_ex[i,j,:] = np.cumsum(
                    np.var(np.dot(h_by_task[t2], comp.T), axis=0))
                var_ex_mis[i,j,:] = np.cumsum(
                    np.var(np.dot(h_by_task[t2], comp[::-1].T), axis=0))

            # Now do control var ex for this activity
            np.random.seed(0)
            rc = scipy.stats.special_ortho_group(total_n_pcs)
            cov = np.cov(h_by_task[t1].T)
            for b in range(n_boot):
                rv = rc.rvs()
                rand_var_ex[i, :, b] = np.cumsum(np.diag(rv.T @ cov @ rv))

        # Now: divide variance explained for activity during each task (when
        # projected onto axes defined from other tasks) by total variance 
        # explainable during that task
        var_ex = np.divide(var_ex, 
            np.tile(per_task_vars[np.newaxis,:,:], (self.N_TASKS,1,1)))

        rand_var_ex = np.divide(rand_var_ex,
            np.tile(per_task_vars[:,:,np.newaxis], (1,1,n_boot)))


        return var_ex.astype(np.float32), var_ex_mis.astype(np.float32), \
            rand_var_ex.astype(np.float32)

    def compute_variance_alignment_by_epoch(self, h, rules):
        """ 
        Variance alignment computed separately in each epoch (to isolate for 
        epoch-specific variation task by task).

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
        Returns: 
            var_ex (list of np.ndarray) -- variance alignment for all task
                pairs, and during each task epoch; each list element is 
                np.ndarray of shape (N_TASKS x N_TASKS x N_TOTAL_PCs), where 
                N_TOTAL_PCS is determined on a task-by-task and epoch-by-epoch
                basis
        """
        var_ex, var_ex_mis, rand_var_ex = [], [], []
        for b in list(self.epoch_bounds.values())[2:]:
            h_rel = h[:, b, :]
            ve, vem, rve = self.compute_variance_alignment(h_rel, rules)
            var_ex.append(ve)
            var_ex_mis.append(vem)
            rand_var_ex.append(rve)

        return var_ex, var_ex_mis, rand_var_ex

    def compute_variance_alignment_by_epoch_no_stim_comp(self, h, rules, sample, w_in):
        """
        Idea: we know that some component of variance will be forced to be 
        shared across tasks, and will be dominant relative to magnitude of 
        stimulus-independent variation in network activity; remove this
        component, either by removing the component of activity found along
        stimulus input dimensions or by doing PCA during first 2 timesteps of
        sample period to identify top few (2?) vectors that are shared, 
        removing those, then performing the analysis.
        """


        '''
        # Subset trials based on sample stimulus ID; take mean
        # of every (unit, timepoint) pair across that group of trials,
        # and subtract at every timepoint from that unit
        for samp in np.unique(sample):

            # Deal separately w/ each stimulus
            samp_trials_0 = np.unique(np.where(sample[:,0] == samp)[0])
            samp_trials_1 = np.unique(np.where(sample[:,1] == samp)[0])
            mean_by_unit_and_t_0 = np.mean(h[samp_trials_0], axis=0)
            mean_by_unit_and_t_1 = np.mean(h[samp_trials_1], axis=0)
            adj_t = self.epoch_bounds['sample'][0]
            h[samp_trials_0, adj_t:,:] -= mean_by_unit_and_t_0[adj_t:,:]
            h[samp_trials_1, adj_t:,:] -= mean_by_unit_and_t_1[adj_t:,:]
        '''

        # Project activity onto the directions that are targeted/manipulated by the 
        # bottom-up input matrix;
        to_remove = []
        for samp in np.unique(sample, axis=0):
            samp_trials = np.unique(np.where(sample == samp)[0])
            act = h[samp_trials]
            to_remove.append(act[:, self.epoch_bounds['sample'],:].mean((0)))

        sample_period_pcs = PCA().fit(np.array(to_remove).reshape((-1, self.N_NEUR)))
        n_pcs_to_remove = np.argwhere(np.cumsum(sample_period_pcs.explained_variance_ratio_) > 0.99)[0][0]
        h_to_change = np.reshape(h, (-1, self.N_NEUR))
        h_to_change = np.dot(h_to_change - sample_period_pcs.mean_, sample_period_pcs.components_.T[:,n_pcs_to_remove:])
        h_to_change = np.dot(h_to_change, sample_period_pcs.components_[n_pcs_to_remove:, :]) + sample_period_pcs.mean_
        h = np.reshape(h_to_change, h.shape)

        #sample_axes_var = 
        #projection_onto_input_dims = np.array([np.dot(np.dot(hi, w_in.T), w_in) for hi in h])
        #print(np.sum(h), np.sum(projection_onto_input_dims), np.sum(h - projection_onto_input_dims))
        #h -= projection_onto_input_dims

        # Perform variance alignment analysis
        var_ex, var_ex_mis, rand_var_ex = \
            self.compute_variance_alignment_by_epoch(h, rules)

        return var_ex, var_ex_mis, rand_var_ex

    def decode_by_task_and_epoch(self, h, rules, y, n_pcs=15):
        """
        Decode y separately in each task's subspace, epoch by epoch.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            y (np.ndarray)   - variable ID to decode (B x 1)
            n_pc (np.ndarray) - number of PCs for each (task, epoch) pair
        Returns:
            decode (np.ndarray) -- np.ndarray of (N_TASKS x T), decoding of y at 
                each timept in each epoch (PCs fit in each epoch)
        """
        # Set up array for recording results
        if len(y.shape) < 2:
            y = y[:, np.newaxis]
        decode = np.zeros((self.N_TASKS, h.shape[1], y.shape[1], n_pcs))

        pca = PCA(n_pcs)

        for i, t in enumerate(np.unique(rules)):

            rel_h = h[np.where(rules == t)[0]]
            rel_y = y[np.where(rules == t)[0]]

            for j, b in enumerate(self.epoch_bounds.values()):
                pca.fit(np.reshape(rel_h[:,b,:], (-1, self.N_NEUR)))
                comp = pca.components_
                mu   = pca.mean_

                for k in range(n_pcs):
                    h_t_bxt_by_n = np.reshape(rel_h, (-1, self.N_NEUR))
                    h_t_k = np.dot(h_t_bxt_by_n - mu, comp.T[:,:k+1])
                    h_t_k = np.reshape(h_t_k, 
                        (rel_h.shape[0], len(b), -1))

                    decode[i, b, :] = self.decode(h_t_k, rel_y).T

        return decode

    def decode_counterfactual(self, h, rules, samples, outputs, n_pcs=30):
        """
        Perform counterfactual decoding analysis - given the stimuli that were
        shown, how well can you decode what the output should've been under
        each other condition.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            samples (np.ndarray) - sample stimulus ID (B x 1)
            n_pc (np.ndarray) - number of PCs for each (task, epoch) pair
        Returns:
            counterfactual_decode (np.ndarray) - np.ndarray of (N_TASKS x 
                N_TASKS x T), decoding of ctfctl output at each timept

        """
        # Set up array for recording results
        counterfactual_decode = np.zeros((self.N_TASKS, self.N_TASKS, 
            len(self.epoch_bounds) - 3, n_pcs//3))
        ctfctl_overlap = np.zeros((self.N_TASKS, self.N_TASKS))
        n_o = len(np.unique(outputs))
        
        # Reduce in dimensionality as needed for each epooch
        pca = PCA(n_pcs)

        # For each pair of tasks (X, Y), take all trials where
        # task was X, and assign to it the label under Y; then try
        # to decode that output from the activity of X
        t0 = time.time()
        for k, b in enumerate(list(self.epoch_bounds.values())[3:]):
            h_b = h[:, b[5:],:]

            for i, t1 in enumerate(np.unique(rules)):
                print(i, f"{str(datetime.timedelta(seconds=int(round(time.time() - t0, 0))))}")
                h_t1_b = h_b[np.where(rules == t1)[0]]

                t1_labels_oh = np.eye(n_o)[outputs[np.where(rules == t1)[0]]-1]
                t1_labels_oh = t1_labels_oh[:, np.newaxis, :]

                # Reduce in dimensionality to n_pcs, then take only top 
                # k components in 1:n_pcs, do decode on those
                pca.fit(np.reshape(h_t1_b, (-1, self.N_NEUR)))
                comp = pca.components_
                mu   = pca.mean_

                for j, t2 in enumerate(np.unique(rules)):
                    if i == j:
                        o_t2 = outputs[np.where(rules == t1)[0]]
                    else:
                        # Assemble counterfactual outputs
                        o_t2 = self.ctfctl_y[np.where(rules == t1)[0],int(t2)]

                    h_t_bxt_by_n = np.reshape(h_t1_b, (-1, self.N_NEUR))

                    for p in range(0, n_pcs, 3):
                    
                        h_t_k = np.dot(h_t_bxt_by_n - mu, comp.T[:,:p+1])
                        h_t_k = np.reshape(h_t_k, (h_t1_b.shape[0], h_t1_b.shape[1], p+1))

                        # Decode counterfactual output from activity through time
                        counterfactual_decode[i, j, k, p//3] = self.decode(h_t_k, o_t2).mean()

                    # Obtain level of overlap between labelings given permutation
                    # (only need to do this once, not separately for each epoch)
                    if k == 0:
                        ctfctl_overlap[i,j] = self.decode(t1_labels_oh, o_t2)

        return counterfactual_decode, ctfctl_overlap

    def compute_pairwise_task_angles(self, w_td):
        """
        Compute the angle between each pair of top-down input vectors;
        do this in the dimensionality of the space spanned by those vectors, 
        whatever that is.

        Args:
            w_td (np.ndarray) - weight mat
        Returns:
            angles (np.ndarray) -- np.ndarray of (N_TASKS x N_TASKS), angle 
                between top-down inputs for each pair of tasks.
            angles_fd (np.ndarray) - same as above, but angles in the full 
                space of top-down input vectors
            total_n_pcs (np.ndarray) - # of PCs used for dim red

        """
        angles    = np.zeros((self.N_TASKS, self.N_TASKS))
        angles_fd = np.zeros((self.N_TASKS, self.N_TASKS))
        w_td      = w_td.squeeze()[1:,...]

        pca_d  = PCA().fit(w_td)
        e_vals = pca_d.explained_variance_ratio_
        total_n_pcs = int(np.sum(e_vals)**2 / np.sum(e_vals**2)) + 1

        w_td_rd = PCA(total_n_pcs).fit_transform(w_td)

        for i in range(w_td.shape[0]):
            for j in range(w_td.shape[0]):
                angles[i,j]    = angle_between(w_td_rd[i], w_td_rd[j])
                angles_fd[i,j] = angle_between(w_td[i], w_td[j])

        return angles, angles_fd, total_n_pcs


    def compute_readout_dimensionality(self, h, rules, outputs, w_out, b_out, n_pcs=100):

        # Subset activity to focus on the test period
        h = h[:, self.epoch_bounds['test'], :]

        # Compute the dimensionality of activity being read out
        all_decodes = np.zeros((self.N_TASKS, n_pcs))
        all_decodes_by_t = np.zeros((self.N_TASKS, h.shape[1], n_pcs))
        
        for i, t in enumerate(np.unique(rules)):

            h_t = h[np.where(rules == t)[0]]
            o_t = outputs[np.where(rules == t)[0]]

            # PCA on all activity
            pca = PCA(n_pcs).fit(np.reshape(h_t, (-1, self.N_NEUR)))
            comp = pca.components_
            mu = pca.mean_
            for k in range(n_pcs):
                h_t_bxt_by_n = np.reshape(h_t, (-1, self.N_NEUR))
                h_t_k = np.dot(h_t_bxt_by_n - mu, comp.T[:,:k+1])
                h_t_k = np.dot(h_t_k, comp[:k+1, :]) + mu
                h_t_k = np.reshape(h_t_k @ w_out[:,1:] + b_out[:,1:], 
                    (h_t.shape[0], h_t.shape[1], -1))
                all_vals = scipy.stats.mode(np.argmax(h_t_k, axis=-1).T+1)[0]
                all_decodes[i, k] = np.sum(all_vals == o_t)/len(o_t)

            # PCA timepoint by timepoint
            for j in range(h.shape[1]):
                h_t_j = h_t[:,j,:].squeeze()
                pca = PCA(n_pcs).fit(h_t_j)
                comp = pca.components_
                mu = pca.mean_
                for k in range(n_pcs):
                    h_t_k = np.dot(h_t_j - mu, comp.T[:,:k+1])
                    h_t_k = np.dot(h_t_k, comp[:k+1, :]) + mu
                    h_t_k = np.reshape(h_t_k @ w_out[:,1:] + b_out[:,1:], 
                        (h_t_j.shape[0], -1))
                    all_vals = np.argmax(h_t_k, axis=-1) + 1
                    all_decodes_by_t[i, j, k] = np.sum(all_vals == o_t)/len(o_t)


        return all_decodes, all_decodes_by_t

    def compute_subspace_specialization(self, h, rules, outputs, samples, n_pcs=20):

        # Set up recording of results (TASKS x TIME)
        dec_shape = (self.N_TASKS, 
                     (h.shape[1] - self.epoch_bounds['sample'][0]) // 3 + 1,
                     n_pcs//2)
        s0_decode, s1_decode, o_decode = np.zeros(dec_shape), \
            np.zeros(dec_shape), np.zeros(dec_shape)

        h   = h[:, self.epoch_bounds['sample'][0]:, :]
        pca = PCA(n_pcs)

        # Walk through tasks, do PCA timepoint-by-timepoint,
        # and then, for each k in (1, N_COMP), decode the relevant info
        for i, t in enumerate(np.unique(rules)):
            h_t = h[np.where(rules == t)[0]]
            s_t = samples[np.where(rules == t)[0]]
            o_t = outputs[np.where(rules == t)[0]]
            for j in range(0, h.shape[1], 3):
                pca.fit(h_t[:,j,:].squeeze())
                comp = pca.components_
                mu   = pca.mean_
                for k in range(0,n_pcs,2):
                    h_t_k = np.dot(h_t[:,j,:].squeeze() - mu, comp.T[:,:k+1])
                    s0_decode[i, j//3, k//2] = self.decode(h_t_k[:,np.newaxis,:], 
                        s_t[:,0])
                    s1_decode[i, j//3, k//2] = self.decode(h_t_k[:,np.newaxis,:], 
                        s_t[:,1])
                    o_decode[i, j//3, k//2] = self.decode(h_t_k[:,np.newaxis,:],
                        o_t)

        return s0_decode, s1_decode, o_decode

    def compute_output_separation_subpopulations(self, h, rules, outputs, w_out):

        h   = h[:, self.epoch_bounds['test'][0]:, :]
        decisions = np.unique(outputs)

        # Set up recording of results (TASKS x TASKS x N_DECISIONS x N_NEUR)
        subpop_sc = np.zeros((self.N_TASKS, len(decisions), self.N_NEUR))

        for i, t in enumerate(np.unique(rules)):

            # Take activity and outputs for this task, and rank units by 
            # how specifically they drive any of the outputs (assess via
            # projection onto outputs separately for each of the tasks)
            h_t = h[np.where(rules == t)[0]]
            o_t = outputs[np.where(rules == t)[0]]

            # Compute output specificity
            for o1 in decisions:

                # Select all trials where output was o1 (if no such trials for this
                # task, just store some zeros and move along)
                rel_trials = np.where(o_t == o1)[0]
                if len(rel_trials) == 0:
                    continue

                # If there were some trials w/ output = o1, multiply them onto 
                # each of the output vectors
                h_t_flat = np.reshape(h_t[rel_trials], (-1, self.N_NEUR))
                o_proj = np.array([np.multiply(h_t_flat, w_out[:,j].T).mean(0) for j in decisions])

                # Take mean projection onto other outptus and subtract
                n_o1 = np.setdiff1d(decisions, [o1]) - 1
                marginal_gains = o_proj[o1 - 1,:] - o_proj[n_o1, :].mean(0)

                # Store marginal gains for this task/output pair for every unit
                # (how much the activity of this unit dials up the activity of 
                # the correct output relative to its impact on the other outputs)
                subpop_sc[i, o1 - 1, :] = marginal_gains

        sc_corrs = np.zeros((self.N_TASKS, self.N_TASKS, len(decisions)))
        for i, t1 in enumerate(np.unique(rules)):
            sc_t1 = subpop_sc[i]
            for j, t2 in enumerate(np.unique(rules)):
                sc_t2 = subpop_sc[j]
                for k, o in enumerate(decisions):
                    sc_t1_o = sc_t1[k] 
                    sc_t2_o = sc_t2[k]
                    sc_corrs[i,j,k] = scipy.stats.spearmanr(sc_t1_o, sc_t2_o)[0]


        # Convert into selectivity indices: 
        # ATD: across-task difference for same output pairs (is projection of this unit onto
        # each of the outputs consistent across tasks? High values = task-specific)
        # WTD: within-task difference for different output pairs (low values = task-specific, 
        # high values = output-specific)
        atd = np.zeros(self.N_NEUR)
        wtd = np.zeros(self.N_NEUR)
        for t in range(subpop_sc.shape[0]):
            for o in range(subpop_sc.shape[1]):

                other_t = np.setdiff1d(np.arange(subpop_sc.shape[0]), [t])
                other_o = np.setdiff1d(np.arange(subpop_sc.shape[1]), [o])

                # ATD: difference between output during 
                # this task and output during every other task
                this_o_all_t = subpop_sc[:,o,:].squeeze()
                this_t_all_o = subpop_sc[t,:,:].squeeze()
                this_o_this_t = this_o_all_t[t]
                this_o_other_t = this_o_all_t[other_t]
                this_t_other_o = this_t_all_o[other_o]

                atd += np.abs((this_o_this_t - this_o_other_t).sum(0))
                wtd += np.abs((this_o_this_t - this_t_other_o).sum(0))


        tai = (wtd - atd) / (wtd + atd)


        return sc_corrs, subpop_sc, tai

    def compute_output_separation_subpopulations_old(self, h, rules, outputs, w_out):

        h   = h[:, self.epoch_bounds['test'][0]:, :]
        decisions = np.unique(outputs)

        # Set up recording of results (TASKS x TASKS x N_DECISIONS x N_NEUR)
        subpop_sc = np.zeros((self.N_TASKS, len(decisions), self.N_NEUR))


        for i, t in enumerate(np.unique(rules)):

            # Take activity and outputs for this task, and rank units by 
            # how specifically they drive any of the outputs (assess via
            # projection onto outputs separately for each of the tasks)
            h_t = h[np.where(rules == t)[0]]
            o_t = outputs[np.where(rules == t)[0]]

            # Compute output specificity
            for o1 in decisions:

                # Select all trials where output was o1 (if no such trials for this
                # task, just store some zeros and move along)
                rel_trials = np.where(o_t == o1)[0]
                if len(rel_trials) == 0:
                    continue

                # If there were some trials w/ output = o1, multiply them onto 
                # each of the output vectors
                h_t_flat = np.reshape(h_t[rel_trials], (-1, self.N_NEUR))
                o_proj = np.multiply(h_t_flat, w_out[:,o1].T).mean(0)

                # Store mean projection value for this output
                subpop_sc[i, o1 - 1, :] = o_proj

        # Convert into selectivity indices: 
        # ATD: across-task difference for same output pairs (is projection of this unit onto
        # each of the outputs consistent across tasks? High values = task-specific)
        # WTD: within-task difference for different output pairs (low values = task-specific, 
        # high values = output-specific)
        atd = np.zeros(self.N_NEUR)
        wtd = np.zeros(self.N_NEUR)
        for t in range(subpop_sc.shape[0]):
            for o in range(subpop_sc.shape[1]):

                other_t = np.setdiff1d(np.arange(subpop_sc.shape[0]), [t])
                other_o = np.setdiff1d(np.arange(subpop_sc.shape[1]), [o])

                # ATD: difference between output during 
                # this task and output during every other task
                this_o_all_t = subpop_sc[:,o,:].squeeze()
                this_t_all_o = subpop_sc[t,:,:].squeeze()
                this_o_this_t = this_o_all_t[t]
                this_o_other_t = this_o_all_t[other_t]
                this_t_other_o = this_t_all_o[other_o]

                atd += np.abs((this_o_this_t - this_o_other_t).sum(0))
                wtd += np.abs((this_o_this_t - this_t_other_o).sum(0))


        tai = (wtd - atd) / (wtd + atd)

        return subpop_sc, tai


    def compute_ensemble_recruitment_similarity(self, w_td):
        """
        Idea: for every pair of tasks, compute difference between 
        XD biases b/w that pair of tasks for every unit, normalized 
        to the range of that bias (e.g. -1 = most different, 
        +1 = most similar)

        """
        ensemble_sims = np.zeros((self.N_TASKS, self.N_TASKS, self.N_NEUR))
        unit_ranges = np.abs(np.amax(w_td, axis=0) - np.amin(w_td, axis=0))
        for i in np.arange(self.N_TASKS):
            for j in np.arange(self.N_TASKS):
                xd_b_i = w_td[i,:]
                xd_b_j = w_td[j,:]
                norm_rng = np.divide(np.abs(xd_b_i - xd_b_j), unit_ranges)
                ensemble_sims[i,j,:] = norm_rng
                ensemble_sims[j,i,:] = norm_rng

        return ensemble_sims

    def obtain_output_subspace(self, h, rules, outputs):
        """


        """

        # Take activity during test period, Murray-style for outputs
        h_test = h[:, self.epoch_bounds['test'], :]
        h_test_by_o = [h_test[np.where(outputs == i)[0]].mean((0,1)) for i in np.unique(outputs)]
        pca_by_o = PCA().fit(np.array(h_test_by_o))

        # Project
        colors = sns.color_palette("Set2", 8)
        fig, ax = plt.subplots(1)
        for i in np.unique(outputs):
            for j in np.unique(rules):
                rel_t = np.where((outputs == i) & (rules == j))[0]
                if len(rel_t) == 0:
                    continue
                rel_h = h_test[rel_t]
                print(rel_h.shape)
                rel_h = np.mean(rel_h, axis=1)
                rel_h = pca_by_o.transform(rel_h)
                ax.scatter(rel_h[:,-2], rel_h[:,-1], color=colors[i - 1])
        plt.tight_layout()
        fig.savefig("output_subspace.png", dpi=500)
        plt.close(fig)
        print(0/0)

        return


class ResultsAnalyzer:

    def __init__(self, args, fns, net_fns):

        self.args = args
        self.fns = fns 
        self.net_fns = net_fns
        self.training_data, self.trained_data = self.load_data(fns, net_fns)

        self.epoch_bounds   = {'dead'  : [0, 300],
                               'fix'   : [300, 500],
                               'sample': [500,800],
                               'delay' : [800,1800],
                               'test'  : [1800,2100]}
        self.N_TASKS = 8
        self.N_SAMPLE = 8
        self.TRIAL_LEN = 2100
        #self.task_list = ["LeftIndication", "RightIndication", 
        #    "MinIndication", "MaxIndication", "SubtractingLeftRight",
        #    "SubtractingRightLeft", "Summing", "SummingOpposite"]
        self.task_list = ['IndicateRF0', 'IndicateRF1', 
            "MinIndication", "MaxIndication", "SubtractingLeftRight",
            "SubtractingRightLeft", "Summing", "SummingOpposite"]

        return

    def get_colors(self, n):
        colors = plt.cm.jet(np.linspace(0,1,n))
        lightdark_colors = plt.cm.tab20(np.linspace(0, 1, n*2))
        light_colors = lightdark_colors[::2]
        dark_colors = lightdark_colors[1::2]

        return colors, light_colors, dark_colors

    def load_data(self, fns, net_fns):
        """ 
        Load all data from specified files; concatenate together based on the 
        fields found in the first file (# networks along axis 0); return.

        Args: 
            fns (list of str) -- list of filenames to load
            iters (np.ndarray) -- array of iterations seen during training
        Returns:
            training_data (dict of np.ndarray) -- dictionary of msmts during tr.
            trained_data (dict of nd.ndarray) -- dictionary of msmts after tr.
        """
        training_data = defaultdict(list)
        trained_data  = defaultdict(list)
        
        for fn, net_fn in zip(fns, net_fns):
            print(fn, net_fn)
            d = pickle.load(open(fn, 'rb'))
            d1 = pickle.load(open(net_fn, 'rb'))
            for k, v in d.items():
                if len(v) == 0:
                    continue
                # Load in the training related data
                training_data[k].append(v)
                trained_data[k].append(v[-1:])

            trained_data['sample_decoding'] = d1['sample_decoding']
            self.iters = d1['save_iters']

        # Bind all into numpy arrays
        #training_data = {k: np.array(v).squeeze() for k, v in 
        #    training_data.items()}
        #trained_data  = {k: np.array(v).squeeze() for k, v in
        #    trained_data.items()}
        return training_data, trained_data


    def compute_alignment_indices(self, va, va_m, va_r, return_pairs=True):
        """
        Compute alignment index, based on area between variance alignment
        curve and random curve; if positive, divide by area between 
        perfect alignment curve (e.g. for axes defined for that task) and 
        random curve; if negative, normalize by the area btwn random curve 
        and the maximal misalignment curve.

        va: np.ndarray, (N_TASKS x N_TASKS x k)
        va_m: np.ndarray, (N_TASKS x N_TASKS x k)
        va_r: np.ndarray, (N_TASKS x k x N_BOOT)
        """
        
        # If only computing between distinct task pairings
        if return_pairs:
            N_PAIRS = self.N_TASKS * (self.N_TASKS - 1) // 2
            pairs = itertools.combinations(np.arange(self.N_TASKS), 2)
        else:
            # If returning N_TASKS x N_TASKS matrix of variance align.
            N_PAIRS = self.N_TASKS**2
            px, py = np.meshgrid(np.arange(self.N_TASKS), np.arange(self.N_TASKS))
            pairs = zip(px.flatten(), py.flatten())

        vai = np.zeros((N_PAIRS))
        for j, p in enumerate(pairs):
            va_rel = va[p[0], p[1], :]
            va_r_rel = va_r[p[0], :, :]
            va_m_rel = va_m[p[0], p[1], :]
            va_max = va[p[0], p[0], :]
            random_curve = np.mean(va_r_rel, -1)
            auc = np.trapz(va_rel - random_curve, dx=1)
            max_align = np.trapz(va_max - random_curve, dx=1)
            min_align = np.trapz(va_m_rel - random_curve, dx=1)
            if auc < 0:
                vai[j] = auc / min_align
            else:
                vai[j] = auc / max_align

        return vai

    def generate_all_figures(self, to_plot=[4]):
        
        ########################################################################
        # I. Variance alignment analysis (networks differentiate tasks into 
        # distinct subspaces that largely do not share variation during test
        # period)
        ########################################################################
        if 0 in to_plot:
            #self.plot_figure_i()
            self.plot_figure_ia()

        ########################################################################
        # II. Readout dimensionality analysis (decisions are read out from 
        # subspaces of varying dimensionality, depending on task/difficulty;
        # learning consists of a progressive lowering of the dimensionality 
        # required for readout)
        ########################################################################
        if 1 in to_plot:
            self.plot_figure_ii()

        ########################################################################
        # III. Subspace specialization analysis (relevant information is 
        # preferentially found in low-dimensional components of network activity
        # as networks learn to perform these tasks)
        ########################################################################
        if 2 in to_plot:
            self.plot_figure_iii()

        ########################################################################
        # IV. Counterfactual decode analysis (how interchangeable are these 
        # subspaces? The control for this analysis also useful in other contexts:
        # do tasks whose structure is similar, e.g. similar output groupings but
        # possibly different labelings, share variation? Use in variance 
        # alignment analysis.)
        ########################################################################
        if 3 in to_plot:
            self.plot_figure_iv()

        return figures

    def plot_figure_i(self):
        """
        Plots of variance alignment.
        (a) Plot alignment index PDF for non-identical pairs, pooled; 

        Do this for each epoch. Could form one such plot, w/ different-shaded 
        traces overlaid for each training iteration, to show that this changes
        across training. Also want to see this task pair by task pair (e.g. in 
        8-by-8 grid) — to see if tasks where outputs are more shared also share 
        activity more.

        """

        # Generate variance alignment indices
        N_NETS = len(self.training_data['var_alignment_by_epoch'])
        N_TR   = len(self.training_data['var_alignment_by_epoch'][0])
        N_EP   = len(self.training_data['var_alignment_by_epoch'][0][0])
        N_PAIRS = self.N_TASKS * (self.N_TASKS - 1) // 2

        align_inds = np.zeros((N_NETS, N_TR, N_EP, N_PAIRS))
        for NET in range(N_NETS):
            for TR in range(N_TR):
                for EP in range(N_EP):
                    align_inds[NET, TR, EP, :] = self.compute_alignment_indices(
                        self.training_data['var_alignment_by_epoch'][NET][TR][EP],
                        self.training_data['var_alignment_mis_by_epoch'][NET][TR][EP],
                        self.training_data['rand_var_alignment_by_epoch'][NET][TR][EP])

        # Plot PDFs of alignment indices (pool across networks, and
        # separately for each epoch in final training iter.)
        fig, ax = plt.subplots(nrows=1, ncols=N_EP, figsize=(3*N_EP, 3))
        titles = list(self.epoch_bounds.keys())[-3:]
        colors = sns.color_palette("crest", N_EP)
        for i, EP in enumerate(np.arange(align_inds.shape[2])):
            pooled_vai = align_inds[:,-1,EP,:].flatten()
            sns.histplot(data=pooled_vai, stat='probability', ax=ax[i], fill=True, alpha=0.7, linewidth=0, color=colors[i])
            ax[i].set(title=titles[EP], xlabel="Alignment index", ylabel="Probability",xlim=[0., 1.])

        plt.tight_layout()
        fig.savefig("var_align_by_ep_final_iter.png", dpi=500)
        plt.close(fig)

        # Plot PDFs of alignment indices (pool across networks, and
        # separately for each epoch but on same subplot in final training iter.)
        fig, ax = plt.subplots(1, figsize=(8,8))
        labels = ['sample', 'delay', 'test']
        pooled_vai = np.transpose(align_inds, (2,0,1,3)) # EP x NET x TR x PAIRS
        pooled_vai = np.reshape(pooled_vai[:,:,-1,:], (N_EP, -1))
        pooled_df = pd.DataFrame(pooled_vai.T, columns=labels)
        sns.histplot(data=pooled_df, stat='probability', fill=True, palette="crest",  ax=ax, common_norm=False, alpha=.7, linewidth=0)
        ax.set(xlabel='Variance alignment')

        plt.tight_layout()
        fig.savefig("var_align_final_iter.png", dpi=500)
        plt.close(fig)

        # Plot PDFs of alignment indices (pooled across networks, and
        # rendered for each training iteration per epoch-specific
        # subplot)
        fig, ax = plt.subplots(nrows=1, ncols=N_EP, figsize=(3*N_EP, 3))
        #titles = list(self.epoch_bounds.keys())
        tr_col = sns.color_palette("crest", N_TR)
        for EP in range(N_EP):
            pooled_vai = align_inds[:,:,EP,:].squeeze()
            pooled_vai = np.transpose(pooled_vai, (1, 0, 2))
            pooled_vai = np.reshape(pooled_vai, (pooled_vai.shape[0], -1))
            pooled_df = pd.DataFrame(pooled_vai.T, columns=[str(i) for i in self.iters])
            pooled_df = pd.melt(pooled_df)
            pooled_df = pooled_df.rename(columns={'variable': 'Epoch', 'value': 'Alignment index'})
            sns.histplot(data=pooled_df, x='Alignment index', stat='probability', palette="crest", fill=True,  
                    ax=ax[EP], linewidth=0, common_norm=False,y="Epoch", hue='Epoch',legend=False)
            ax[EP].set(title=titles[EP], xlabel='Alignment index', ylabel='Epoch')

        plt.tight_layout()
        fig.savefig("var_align_by_ep_training.png", dpi=500)
        plt.close(fig)

        # Plot matrix of pairwise variance alignment at final training iteration
        # during test period
        aip = []
        for NET in range(N_NETS):
            aip.append(self.compute_alignment_indices(
                self.training_data['var_alignment_by_epoch'][NET][-1][-1],
                self.training_data['var_alignment_mis_by_epoch'][NET][-1][-1],
                self.training_data['rand_var_alignment_by_epoch'][NET][-1][-1],
                False).reshape((self.N_TASKS, self.N_TASKS)))

        aip = np.array(aip).mean(0)
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
        aip[np.diag_indices(self.N_TASKS)] = np.nan
        sns.heatmap(aip, ax=ax[0], square=True, linewidths=1)
        ax[0].set_xticklabels(self.task_list, rotation = 45, ha="right")
        ax[0].set_yticklabels(self.task_list, rotation=0)
        ax[0].set_title("Pairwise task var. alignment indices")
        

        # Same idea, but plot counterfactual label overlap
        cflo = np.array(self.training_data['counterfactual_overlap']).mean(0)
        cflo[np.diag_indices(self.N_TASKS)] = np.nan
        sns.heatmap(cflo, ax=ax[1], square=True, linewidths=1)
        ax[1].set_xticklabels(self.task_list, rotation = 45, ha="right")
        ax[1].set_yticklabels(self.task_list, rotation=0)
        ax[1].set_title("Label overlap")
        fig.suptitle("Label overlap and variance alignment")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig("var_align_pairwise_with_overlap.png", dpi=500)
        plt.close(fig)

        return

    def plot_figure_ia(self):
        """
        Plots of variance alignment (for the modified, no-stim case)
        (a) Plot alignment index PDF for non-identical pairs, pooled; 

        Do this for each epoch. Could form one such plot, w/ different-shaded 
        traces overlaid for each training iteration, to show that this changes
        across training. Also want to see this task pair by task pair (e.g. in 
        8-by-8 grid) — to see if tasks where outputs are more shared also share 
        activity more.

        """

        # Generate variance alignment indices
        N_NETS = len(self.training_data['var_alignment_by_epoch_no_stim'])
        N_TR   = len(self.training_data['var_alignment_by_epoch_no_stim'][0])
        N_EP   = len(self.training_data['var_alignment_by_epoch_no_stim'][0][0])
        N_PAIRS = self.N_TASKS * (self.N_TASKS - 1) // 2

        align_inds = np.zeros((N_NETS, N_TR, N_EP, N_PAIRS))
        for NET in range(N_NETS):
            for TR in range(N_TR):
                for EP in range(N_EP):
                    align_inds[NET, TR, EP, :] = self.compute_alignment_indices(
                        self.training_data['var_alignment_by_epoch_no_stim'][NET][TR][EP],
                        self.training_data['var_alignment_mis_by_epoch_no_stim'][NET][TR][EP],
                        self.training_data['rand_var_alignment_by_epoch_no_stim'][NET][TR][EP])

        # Plot PDFs of alignment indices (pool across networks, and
        # separately for each epoch in final training iter.)
        fig, ax = plt.subplots(nrows=1, ncols=N_EP, figsize=(3*N_EP, 3))
        titles = list(self.epoch_bounds.keys())[-3:]
        colors = sns.color_palette("crest", N_EP)
        for i, EP in enumerate(np.arange(align_inds.shape[2])):
            pooled_vai = align_inds[:,-1,EP,:].flatten()
            sns.histplot(data=pooled_vai, stat='probability', ax=ax[i], fill=True, alpha=0.7, linewidth=0, color=colors[i])
            ax[i].set(title=titles[EP], xlabel="Alignment index", ylabel="Probability",xlim=[0., 1.])

        plt.tight_layout()
        fig.savefig("var_align_by_ep_no_stim_final_iter.png", dpi=500)
        plt.close(fig)

        # Plot PDFs of alignment indices (pool across networks, and
        # separately for each epoch but on same subplot in final training iter.)
        fig, ax = plt.subplots(1, figsize=(8,8))
        labels = ['sample', 'delay', 'test']
        pooled_vai = np.transpose(align_inds, (2,0,1,3)) # EP x NET x TR x PAIRS
        pooled_vai = np.reshape(pooled_vai[:,:,-1,:], (N_EP, -1))
        pooled_df = pd.DataFrame(pooled_vai.T, columns=labels)
        sns.histplot(data=pooled_df, stat='probability', fill=True, palette="crest",  ax=ax, common_norm=False, alpha=.7, linewidth=0)
        ax.set(xlabel='Variance alignment')

        plt.tight_layout()
        fig.savefig("var_align_no_stim_final_iter.png", dpi=500)
        plt.close(fig)

        # Plot PDFs of alignment indices (pooled across networks, and
        # rendered for each training iteration per epoch-specific
        # subplot)
        fig, ax = plt.subplots(nrows=1, ncols=N_EP, figsize=(3*N_EP, 3))
        #titles = list(self.epoch_bounds.keys())
        tr_col = sns.color_palette("crest", N_TR)
        for EP in range(N_EP):
            pooled_vai = align_inds[:,:,EP,:].squeeze()
            pooled_vai = np.transpose(pooled_vai, (1, 0, 2))
            pooled_vai = np.reshape(pooled_vai, (pooled_vai.shape[0], -1))
            pooled_df = pd.DataFrame(pooled_vai.T, columns=[str(i) for i in self.iters])
            pooled_df = pd.melt(pooled_df)
            pooled_df = pooled_df.rename(columns={'variable': 'Epoch', 'value': 'Alignment index'})
            sns.histplot(data=pooled_df, x='Alignment index', stat='probability', palette="crest", fill=True,  
                    ax=ax[EP], linewidth=0, common_norm=False,y="Epoch", hue='Epoch',legend=False)
            ax[EP].set(title=titles[EP], xlabel='Alignment index', ylabel='Epoch')

        plt.tight_layout()
        fig.savefig("var_align_by_ep_no_stim_training.png", dpi=500)
        plt.close(fig)

        # Plot matrix of pairwise variance alignment at final training iteration
        # during test period
        aip = []
        for NET in range(N_NETS):
            aip.append(self.compute_alignment_indices(
                self.training_data['var_alignment_by_epoch_no_stim'][NET][-1][-1],
                self.training_data['var_alignment_mis_by_epoch_no_stim'][NET][-1][-1],
                self.training_data['rand_var_alignment_by_epoch_no_stim'][NET][-1][-1],
                False).reshape((self.N_TASKS, self.N_TASKS)))

        aip = np.array(aip).mean(0)
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
        aip[np.diag_indices(self.N_TASKS)] = np.nan
        sns.heatmap(aip, ax=ax[0], square=True, linewidths=1)
        ax[0].set_xticklabels(self.task_list, rotation = 45, ha="right")
        ax[0].set_yticklabels(self.task_list, rotation=0)
        ax[0].set_title("Pairwise task var. alignment indices")
        

        # Same idea, but plot counterfactual label overlap
        cflo = np.array(self.training_data['counterfactual_overlap']).mean(0)
        cflo[np.diag_indices(self.N_TASKS)] = np.nan
        sns.heatmap(cflo, ax=ax[1], square=True, linewidths=1)
        ax[1].set_xticklabels(self.task_list, rotation = 45, ha="right")
        ax[1].set_yticklabels(self.task_list, rotation=0)
        ax[1].set_title("Label overlap")
        fig.suptitle("Label overlap and variance alignment")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig("var_align_no_stim_pairwise_with_overlap.png", dpi=500)
        plt.close(fig)

        return

    def plot_figure_ii(self):
        """
        Plots of readout dimensionality.

        For each task, plot the curve of readout accuracy as a 
        function of number of components included; should emerge 2 groups 
        (one where accuracy achieved via the readout from low-dim comps; 
        one where accuracy achieved via the readout of activity into higher 
        dimensions, e.g. tasks where information must be combined across RFs).
        """

        # Pull into np array (N_NETs x N_TR x N_TASKS x N_DIM)
        rd_dec = np.array(self.training_data['rd_decodes'])
        accs = np.array(self.training_data['accs']) / 2

        # Push into pandas dataframe:
        rd_dec = np.transpose(rd_dec, (2, 0, 1, 3))
        long_data = []
        first_past_5x = []
        for t in range(rd_dec.shape[0]):
            for n in range(rd_dec.shape[1]):
                for tr in range(rd_dec.shape[2]):
                    rd = [rd_dec[t,n,tr,0]]
                    rd.extend(np.diff(rd_dec[t,n,tr,:]))
                    rd = np.array(rd)
                    if np.sum(rd) == 0:
                        continue
                    rd = int(np.sum(rd)**2 / np.sum(rd**2)) + 1
                    fp = np.where(rd_dec[t,n,tr,:] > 0.625)[0]
                    if len(fp) > 0:
                        first_past_5x.append([self.task_list[t], n, self.iters[tr], accs[n,tr,t], rd, fp[0]])
                    else:
                        first_past_5x.append([self.task_list[t], n, self.iters[tr], accs[n,tr,t], rd, np.nan])
                    for k in range(rd_dec.shape[3]):
                        long_data.append([self.task_list[t], n, self.iters[tr], k, rd_dec[t, n, tr, k]])
        long_data = pd.DataFrame(long_data, 
            columns=["Task", "Net", "Epoch", "Rank", "Decision decode"])

        # Plot for final Epoch curve for each task of decode vs. rank
        fig, ax = plt.subplots(1, figsize=(5,5))

        # Plot separately by low_d vs. high_d readout
        low_d = long_data[long_data['Task'].isin(self.task_list[:4])]
        high_d = long_data[long_data['Task'].isin(self.task_list[4:])]
        ax.axhline(y=1.0/8, linestyle='--', color='grey', label=f'Chance ({100/8:.2f}%)')
        sns.lineplot(data=low_d[low_d['Epoch'] == self.iters[-1]], x='Rank', y='Decision decode', hue='Task', palette='crest')
        sns.lineplot(data=high_d[high_d['Epoch'] == self.iters[-1]], x='Rank', y='Decision decode', hue='Task', palette='flare')
        
        plt.tight_layout()
        fig.savefig("readout_dim.png", dpi=500)
        plt.close(fig)

        # Take dimension where decode first breaks 62.5% (5x chance); 
        # Form scatter plot of accuracy on each (task,net) at each tr iter
        # fit correlation line
        first_past_5x = pd.DataFrame(first_past_5x, 
            columns=["Task", "Net", "Epoch", "Accuracy", "Rank", "Rank (decode > 0.625)"])
        fig, ax = plt.subplots(nrows=2, ncols=self.N_TASKS//2, figsize=(1.5*self.N_TASKS, 5))
        task_cols = sns.color_palette('crest', 4)
        task_cols.extend(sns.color_palette('flare', 4))
        rows = [0, 0, 1, 1, 0, 0, 1, 1]
        cols = [0, 1, 0, 1, 2, 3, 2, 3]
        for i, t in enumerate(self.task_list):
            axi = ax[rows[i], cols[i]]
            sns.scatterplot(data=first_past_5x[first_past_5x['Task'] == t], x='Accuracy', y='Rank', color=task_cols[i], ax=axi)
            #sns.lineplot(data=first_past_5x[first_past_5x['Task'] == t], x='Accuracy', y='Rank', color=task_cols[i], ax=axi)
            axi.set(title=t)
        plt.tight_layout()
        fig.savefig("readout_dimension_estimate_through_training.png", dpi=500)
        plt.close(fig)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
        epoch_net_combo = first_past_5x['Net'].astype(str) + first_past_5x['Epoch'].astype(str)
        first_past_5x['units'] = epoch_net_combo

        # Plot traces
        sns.lineplot(data=first_past_5x, x='Accuracy', y='Rank (decode > 0.625)', ax=ax[0], hue='Epoch', palette='flare', units='units', estimator=None, alpha=0.7)
        sns.move_legend(
            ax[0], "center left",
            bbox_to_anchor=(1, 0.5),
            ncol=1,
            frameon=False,
        )
        ax[0].set(title='Readout dimension vs. accuracy (per net/tr. iter.)')

        # Compute and plot distribution of correlations
        corrs = []
        for unit in first_past_5x.units.unique():
            rel_units = first_past_5x[first_past_5x.units == unit]
            rel_units = rel_units.dropna(subset=['Rank (decode > 0.625)'])
            corr = scipy.stats.spearmanr(rel_units.Accuracy, rel_units['Rank (decode > 0.625)'])[0]
            corrs.append(corr)
        sns.histplot(corrs, ax=ax[1], palette='flare')
        ax[1].set(xlabel=r"Correlation (Spearman's $\rho$)", title=r"Readout dim. vs. accuracy (Spearman's $\rho$)")
        plt.tight_layout()
        fig.savefig("readout_first_above5xchance_through_training.png", dpi=500)
        plt.close(fig)


        return

    def plot_figure_iii(self):
        """
        Plots of subspace specialization.
        
        For each task, plot the level of decoding of each of the variables 
        through time; do this for lowest-D and highest-D decoding (idea: 
        if the subspace is specialized, more information about the stimuli 
        necessary for its decision should be available in the low-d components, 
        e.g. the first 4 tasks)
        """

        # Obtain data (NETS x TR x TASKS x TIME x NDIM)
        s0_dec = np.array(self.training_data['s0_decodes'])
        s1_dec = np.array(self.training_data['s1_decodes'])
        o_dec = np.array(self.training_data['o_decodes'])

        print(s0_dec.shape)
        #self.iters = [self.iters[0], self.iters[-1]]

        long_data = []

        # Make into dataframe
        for NET in range(s0_dec.shape[0]):
            for TR in range(s0_dec.shape[1]):
                for TASK in range(s0_dec.shape[2]):
                    for TIME in range(s0_dec.shape[3]):
                        for NDIM in range(s0_dec.shape[4]):
                            s0 = s0_dec[NET, TR, TASK, TIME, NDIM]
                            s1 = s1_dec[NET, TR, TASK, TIME, NDIM]
                            o = o_dec[NET, TR, TASK, TIME, NDIM]
                            long_data.append([NET, self.iters[TR], self.task_list[TASK], TIME, NDIM+1, s0, 'RF 0 stim.'])
                            long_data.append([NET, self.iters[TR], self.task_list[TASK], TIME, NDIM+1, s1, 'RF 1 stim.'])
                            long_data.append([NET, self.iters[TR], self.task_list[TASK], TIME, NDIM+1, o, 'Decision'])

        long_data = pd.DataFrame(long_data, columns=["Net", 
            "Epoch", "Task", "Time", "Rank", "Decode", "Variable"])

        # Adjust time to reflect ms in task
        #long_data['Time'] = long_data['Time'] * 40 + 500
        long_data['Time'] = long_data['Time'] * 60 + 500

        # Rank
        #rank_range = [2, 4, 8, 16]
        long_data['Rank'] = long_data['Rank'] * 2
        rank_range = [2,4,8]
        print(np.unique(long_data['Rank']))

        # Make figure (final tr epoch)
        palettes = ['crest', 'crest', 'crest']
        for i in range(self.N_TASKS):
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3))
            t = self.task_list[i]
            ld = long_data[(long_data['Task'] == t) & (long_data['Epoch'] == self.iters[-1]) & (long_data['Rank'].isin(rank_range))]
            sns.lineplot(data=ld[ld['Variable'] == 'RF 0 stim.'], x='Time', y='Decode', hue='Rank', palette=palettes[0], ax=ax[0])
            sns.lineplot(data=ld[ld['Variable'] == 'RF 1 stim.'], x='Time', y='Decode', hue='Rank', palette=palettes[1], ax=ax[1])
            sns.lineplot(data=ld[ld['Variable'] == 'Decision'], x='Time', y='Decode', hue='Rank', palette=palettes[2], ax=ax[2])
            ax[0].set(title="RF 0 decode", ylim=[0.15, 1.])
            ax[1].set(title="RF 1 decode", ylim=[0.15, 1.])
            ax[2].set(title="Decision decode", ylim=[0.15, 1.])
            sns.move_legend(ax[0], bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False)
            sns.move_legend(ax[1], bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False)
            sns.move_legend(ax[2], bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False)
            fig.suptitle(t)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(f"specialization_task={t}.png", dpi=500)
            plt.close(fig)

        # Make figure: for each task, plot decode through time for first 5 ranks of each variable
        # (one subplot per var), with traces colored by rank and styled by 
        # Make figure (final tr epoch)
        rank_range = [1,2,4,8]
        rank_range = [2, 4, 8]
        palettes = ['crest', 'crest', 'crest']
        for i in range(self.N_TASKS):
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3))
            t = self.task_list[i]
            ld = long_data[(long_data['Task'] == t) & (long_data['Rank'].isin(rank_range)) & (long_data['Epoch'].isin([self.iters[0], self.iters[-1]]))]
            sns.lineplot(data=ld[ld['Variable'] == 'RF 0 stim.'], x='Time', y='Decode', hue='Rank', style='Epoch', palette=palettes[0], ax=ax[0])
            sns.lineplot(data=ld[ld['Variable'] == 'RF 1 stim.'], x='Time', y='Decode', hue='Rank', style='Epoch',palette=palettes[1], ax=ax[1])
            sns.lineplot(data=ld[ld['Variable'] == 'Decision'], x='Time', y='Decode', hue='Rank', style='Epoch', palette=palettes[2], ax=ax[2])
            ax[0].set(title="RF 0 decode", ylim=[0.15, 1.])
            ax[1].set(title="RF 1 decode", ylim=[0.15, 1.])
            ax[2].set(title="Decision decode", ylim=[0.15, 1.])
            sns.move_legend(ax[0], bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False)
            sns.move_legend(ax[1], bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False)
            sns.move_legend(ax[2], bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False)
            fig.suptitle(t)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(f"specialization_training_task={t}.png", dpi=500)
            plt.close(fig)
        
        return

    def plot_figure_iv(self):
        """
        Plots of counterfactual decode.

        Plot matrix of decoding accuracies in the delay and test periods, 
        divided by the level of label overlap; do this for the lowest-D 
        and highest-D decoding, and do it for the delay and test epochs; 
        if the subspaces are specialized, then cross-task output decode 
        should be low in low-dimensional components (w/ high-dimensional 
        decode allowing for possibility that information is still present).

        Interpetation of dimensions: 
        In NET [0], at training iteration [1], using the activity during 
        trials where the network was cued to perform task [2], if you try
        to decode what the output would have been if it was a trial of 
        task [3], during epoch [4], from the first [5] PCs of that task's 
        activity....
        """

        # Collect counterfactual decode data

        # N_NETS x N_TR x N_TASKS x N_TASKS x N_EP x N_COMP
        ctfctl_dec = np.array(self.training_data['counterfactual_decode'])

        # N_NETS x N_TASKS x N_TASKS
        ctfctl_overlap = np.array(self.training_data['counterfactual_overlap'])

        eps = list(self.epoch_bounds.keys())[-2:]
        

        # Make lineplot: counterfactual decode for one pairing of tasks 
        # as a function of number of PCs
        tasks_to_plot = [0, 3, 5, 6, 7]
        iters_to_plot = [0, -1]

        colors = sns.color_palette('Set2').as_hex()
        
        for t in tasks_to_plot:
            for i in iters_to_plot:
                sns.set_palette("Set2")
                fig, ax = plt.subplots(1, figsize=(5,3))
                yes_t = [t]
                no_t = np.setdiff1d(np.arange(ctfctl_dec.shape[2]), t)
                print(yes_t, no_t)
                d0 = ctfctl_dec[:,i,yes_t,yes_t,-1,:].squeeze() #/ ctfctl_dec[:,-1,-2,-2,-1,:].squeeze()
                d1 = ctfctl_dec[:,i,no_t,yes_t,-1,:].squeeze() #/ ctfctl_dec[:,-1,0,0,-1,:].squeeze()
                lines = ax.plot(np.arange(0, d0.shape[1]*3, 3), d0.mean(0), linewidth=2)
                l1s = ax.plot(np.arange(0, d1.shape[-1]*3, 3), d1.mean(0).T, color='black')

                # Special additions here
                if t == 5:
                    l1s[4].set_color(colors[1])
                if t == 0:
                    l1s[0].set_color(colors[1])
                if t == 3:
                    l1s[2].set_color(colors[1])


                lines.extend(l1s)
                labels = [self.task_list[t]]
                labels.extend([self.task_list[n] for n in no_t])
                ax.legend(lines, labels)
                ax.set(ylim=[0.15, 1.],
                       xlabel='# PCs',
                       ylabel='Decode')
                sns.move_legend(ax, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False)
                fig.tight_layout()
                
                fig.savefig(f"ctfctl_dec_lineplot_task={t}_iter={i}.png", dpi=500)
                plt.close(fig)
        return
        # Assemble data into pandas format
        '''
        long_data = []
        
        for NET in range(ctfctl_dec.shape[0]):
            for TR in range(ctfctl_dec.shape[1]):
                for T0 in range(ctfctl_dec.shape[2]):
                    for T1 in range(ctfctl_dec.shape[3]):
                        ovlp = ctfctl_overlap[NET, T0, T1]
                        for EP in range(ctfctl_dec.shape[4]):
                            for COMP in range(ctfctl_dec.shape[5]):
                                d = ctfctl_dec[NET, TR, T0, T1, EP, COMP]
                                dat = [NET, self.iters[TR], self.task_list[T0], 
                                    self.task_list[T1], eps[EP], COMP, d, ovlp]
                                long_data.append(dat)
        
        long_data = pd.DataFrame(long_data, columns=['Net', 'Epoch', 
            'Task (act.)', 'Task (lab.)', 'Task period', 'Rank', 
            'Decode', 'Overlap'])

        # Plot matrix of counterfactual / overlap
        ld_0 = long_data[long_data['Epoch'] == self.iters[0]]
        ld_f = long_data[long_data['Epoch'] == self.iters[-1]]
        ld_0_d = ld_0[ld_0['Task period'] == eps[0]]
        ld_0_t = ld_0[ld_0['Task period'] == eps[1]]
        ld_f_d = ld_f[ld_f['Task period'] == eps[0]]
        ld_f_t = ld_f[ld_f['Task period'] == eps[1]]
        data = [[ld_0_d, ld_0_t], [ld_f_d, ld_f_t]]
        for i in len(data):
        '''
        
        for i in range(2):
            for j, ep in enumerate([0, len(self.iters)-1]):
                fig, ax = plt.subplots(1, figsize=(6, 4))
                d = ctfctl_dec[:,ep,:,:,i,:3].squeeze()
                d[np.where(d == 1)] = np.nan
                d /= d.max(1)[:,np.newaxis,:,:]
                d = d.mean((0,-1))
                #d = ctfctl_overlap.mean(0)#np.divide(d, ctfctl_overlap).mean(0)
                sns.heatmap(d, ax=ax, square=True, linewidths=1)
                ax.set_xticklabels(self.task_list, rotation = 45, ha="right")
                ax.set_yticklabels(self.task_list, rotation=0)
                fig.suptitle("Norm. counterfactual decode")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig.savefig(f"ctfctl_dec_epoch={self.iters[ep]}_taskperiod={eps[i]}.png", dpi=500)
                plt.close(fig)
        return

    def plot_figure_v(self):

        # Hierarchically cluster and plot (define ordering based on values just during
        # task 1, see if this applies to all other tasks)
        clustering = KMeans(9).fit(subpop_sc).labels_


        # If plotting: sort output weight matrix by clustering, plot them, 
        # and save
        fig, ax = plt.subplots(1, figsize=(5,5))
        sorted_c = np.argsort(clustering)
        imshow_kwargs = {"interpolation": 'none',
                         "aspect"       : 'auto',
                         "cmap"         : 'viridis'}
        sns.heatmap(subpop_sc[sorted_c,:])

        ax.set(title='Output specificity clustering',
               ylabel='Hidden unit #')
        plt.tight_layout()
        fig_fn = f"{self.fig_path}_output_projection_clustering.png"
        fig.savefig(fig_fn, dpi=300)
        plt.close(fig) 

        # Do the same for subpop_sc_all
        subpop_sc_all = subpop_sc_all[:,:,nzrows]
        

        # Also: do PCA, plot twice -- once colored by task max, once colored by output max
        # First: take the mean on dimensions where the column is not all 0s
        all_zero_cols = np.where(np.sum(subpop_sc_all, axis=(2)) == 0)
        subpop_sc_all /= sc_m[np.newaxis,np.newaxis,:]
        subpop_sc_all[all_zero_cols[0],all_zero_cols[1],:] = np.nan

        output_pref = np.argmax(np.nanmedian(subpop_sc_all,axis=0),axis=0)
        task_pref   = np.argmax(np.nanmedian(subpop_sc_all,axis=1),axis=0)

        pca = PCA(2).fit_transform(subpop_sc)
        df = pd.DataFrame({'PC1': pca[:,0], 'PC2': pca[:,1], 
            'Output pref.': self.output_list[output_pref], 'Task pref.': self.task_list[task_pref]})
        print(pd.unique(df['Output pref.']))
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Output pref.', palette="Set2", ax=ax[0], linewidth=0, hue_order=self.output_list)
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Task pref.', palette="Set2", ax=ax[1],linewidth=0, hue_order=self.task_list)

        sns.move_legend(ax[0], bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False)
        sns.move_legend(ax[1], bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, frameon=False)

        plt.tight_layout()
        fig_fn = f"{self.fig_path}_output_projection_pca_clustering.png"
        fig.savefig(fig_fn, dpi=300)
        plt.close(fig) 


def make_taskwise_figure():

    fig = plt.figure(figsize=(12, 6))
    # Set up row/col spacing
    num_rows = 2
    num_cols = 4
    row_height = 3
    space_height = 5

    num_sep_rows = lambda x: int((x-1)/2)
    grid = (row_height*num_rows + space_height*num_sep_rows(num_rows), num_cols)

    ax = []

    for ind_row in range(num_rows):
        for ind_col in range(num_cols):
            grid_row = row_height*ind_row + space_height*num_sep_rows(ind_row+1)
            grid_col = ind_col

            ax += [plt.subplot2grid(grid, (grid_row, grid_col), rowspan=row_height)]


    plt.subplots_adjust(hspace=0.4)


    return fig, ax

def unit_vector(vector):
    """ 
    Normalizes input vector.
    """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors v1 and v2.

    Args:
        v1 (np.ndarray) - vector, (n x 1)
        v2 (np.ndarray) - vector, (n x 1)
    Returns:
        angle (float) - angle between v1 and v2, rad.
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if angle > np.pi / 2:
        angle = np.pi - angle
    return np.degrees(angle)

def get_fns(data_path=args.data_path):
    return glob.glob(data_path + "*_tr.pkl")


if __name__ == "__main__":
    fns = get_fns()
    if not os.path.exists(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.exists(args.analysis_save_path):
        os.makedirs(args.analysis_save_path)

    task_titles = ["LeftIndication", "RightIndication", \
        "MinIndication", "MaxIndication", "SubtractingLeftRight",
        "SubtractingRightLeft", "Summing", "SummingOpposite"]

    results_fns = []
    net_fns = []
    print(fns)

    for fn in fns:
        print(fn)

        net_analyzer = NetworkAnalyzer(fn, args)
        if not hasattr(net_analyzer, 'h'):
            results_fns.append(net_analyzer.analysis_save_f)
            net_fns.append(fn)
            continue

        # Perform specified analyses
        analyses = [25, 30]#list(range(23,40)) + ["23a"]
        results, results_fn = net_analyzer.perform_all_analyses(analyses)
        results_fns.append(results_fn)
        net_fns.append(fn)


    # Perform summary analyses
    ra = ResultsAnalyzer(args, results_fns, net_fns)
    ra.generate_all_figures([3])
    

    