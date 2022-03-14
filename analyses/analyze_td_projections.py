import compress_pickle as pickle, glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
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
import scipy.stats
import scipy.ndimage as sim

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
parser.add_argument('--data_path', type=str, default='./results/for_activity_analysis/')
parser.add_argument('--figure_save_path', type=str, default='./results/for_activity_analysis/figures/')
parser.add_argument('--analysis_save_path', type=str, default='./results/for_activity_analysis/analysis/')
parser.add_argument('--N_PCS', type=int, default=5)
parser.add_argument('--N_MAX_PCS', type=int, default=5)
parser.add_argument('--k_folds', type=int, default=4)
parser.add_argument('--do_plot', type=str2bool, default=True)
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
        if data == 0:
            return
        self.h, self.w_td, self.w_rec, self.w_out, self.acc, \
            self.task_ids, self.sample, self.labels = data

        # Extract some key elements/shapes
        self.N_TASKS   = len(np.unique(self.task_ids))
        self.N_SAMPLES = len(np.unique(self.sample))
        self.N_NEUR    = self.h.shape[-1]
        self.T         = self.h.shape[1]

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
            return 0

        data = pickle.load(open(fn, 'rb'))
        alt_data = pickle.load(open(fn[:-7] + ".pkl", 'rb'))
        if len(data.keys()) == 0:
            return 0
        h = data['train_acts'][0][-1,...]
        w_td = data['top_down0_w:0'][-1] @ data['top_down1_w:0'][-1]
        w_rec = data['rnn_w:0'][-1]
        w_out = data['policy_w:0'][-1]
        self.b_out = data['policy_b:0'][-1]
        self.w_in = data['bottom_up_w:0'][-1][:-3,:]

        acc = np.mean(np.array(alt_data['task_accuracy'])[:, -25:, :], axis=(0,1))

        # Save 
        self.train_acts = data['train_acts'][0]
        self.train_rules = data['train_rules'][0]
        self.train_accs = data['train_accs'][0]
        self.train_labels = data['train_labels'][0]
        self.train_samples = data['train_samples'][0]
        #self.train_samples = self.adjust_samples(self.train_rules, 
        #    self.train_samples)
        self.train_w_outs = data['policy_w:0']

        task_ids = self.train_rules
        sample = self.train_samples
        labels = self.train_labels

        return h, w_td, w_rec, w_out, acc, task_ids, sample, labels

    def adjust_samples(self, rules, samples):

        samples_adj = np.zeros(samples.shape[0])

        new_keys = {}
        ct = 0
        for i in np.unique(samples[:,0]):
            for j in np.unique(samples[:,1]):
                samples_adj[np.where((samples[:,0] == i) & (samples[:,1] == j))[0]] = ct
                ct += 1

        return samples_adj

    def adjust_samples_old(self, rules, samples):
        samples_adj = np.zeros(samples.shape[0])

        # A. Take sample of non- 5 or 6 trials as-is
        samples_adj[np.where(rules < 4)[0]] = samples[np.where(rules < 4)[0],0]

        # B. Convert delay-go to category
        dly_go_tr = np.where(rules == 4)[0]
        samples_adj[dly_go_tr] = samples[dly_go_tr,0] // 3

        # C. Convert pro/retro to category combo of both stim.
        pro_ret_tr = np.where(rules >= 5)[0]
        c0 = samples[pro_ret_tr,0] // 3
        c1 = samples[pro_ret_tr,1] // 3
        samples_adj[pro_ret_tr] = 2 * c0 + c1
        return samples_adj

    def get_colors(self):
        colors = plt.cm.jet(np.linspace(0,1,self.N_TASKS))
        lightdark_colors = plt.cm.tab20(np.linspace(0, 1, self.N_TASKS*2))
        light_colors = lightdark_colors[::2]
        dark_colors = lightdark_colors[1::2]

        return colors, light_colors, dark_colors

    def perform_all_analyses(self, to_do=np.arange(40)):

        print(np.where(np.isnan(self.h)))

        results = {}

        # If file already exists, don't overwrite unless specified
        analysis_save_f = f"{self.analysis_path}_analysis.pkl"
        if os.path.exists(analysis_save_f) and not self.args.do_overwrite:
            results = pickle.load(open(analysis_save_f, 'rb'))
            print(analysis_save_f)
            return results, analysis_save_f

        t0 = time.time()

        # 0. Compute task-specificity of projection onto output
        if 0 in to_do:
            out_task_dec = []
            if hasattr(self, 'train_acts') and self.args.do_learning_analysis:
                train_vars = zip(self.train_acts, self.train_w_outs)
                for i, (h_i, w_i) in enumerate(train_vars):
                    o_t_d = self.compute_output_projection(h_i, self.task_ids, 
                        w_i)
                    out_task_dec.append(o_t_d)
            else:
                o_t_d = self.compute_output_projection(self.h, self.task_ids, 
                    self.w_out)
                out_task_dec.append(o_t_d)
            results['out_task_dec'] = np.array(out_task_dec).squeeze()

        # 1. Compute clustering of output weights
        if 1 in to_do:
            output_clustering = self.cluster_output_weights(self.w_out)
            results['output_clustering'] = output_clustering

        ########################################################################
        # Identifying context-dependent and -independent dimensions of activity
        ########################################################################
        # 2. Identify context-PCs within full network activity space
        if 2 in to_do:
            context_pcs, ctx_pcs_exp_var = self.identify_context_subspace(self.h)
            results['context_pcs_exp_var'] = ctx_pcs_exp_var

        # 3. Identify context-independent PCs within full network activity space
        if 3 in to_do:
            noncontext_pcs, nctx_pcs_exp_var = \
                self.identify_context_independent_subspace(self.h, context_pcs)
            results['noncontext_pcs_exp_var'] = nctx_pcs_exp_var
        
        # 4. Identify axes of variation for each task separately; determine N_PCS
        # per task that explains >90 pct of variance
        if 4 in to_do:
            per_task_pcs, explained_var = self.identify_pcs_per_task(self.h)
            results['per_task_exp_var'] = explained_var

        ########################################################################
        # Decoding context from activity
        ########################################################################
        # 5. Decode context from network activity (raw)
        if 5 in to_do:
            print(5)
            context_decode_raw = self.decode(self.h, self.task_ids,
                save_fn="context_decode_raw", do_reduce=True)
            results['context_decode_raw'] = context_decode_raw

        # 6. Decode context from activity (task-dependent subspace)
        if 6 in to_do:
            print(6)
            context_decode_proj_td = self.decode(self.h, self.task_ids,
                context_pcs, save_fn="context_decode_task_dependent_subspace")
            results['context_decode_proj_td'] = context_decode_proj_td
        
        # 7. Decode context from activity (task-independent subspace)
        if 7 in to_do:
            print(7)
            context_decode_proj_ti = self.decode(self.h, self.task_ids,
                noncontext_pcs, save_fn="context_decode_task_independent_subspace")
            results['context_decode_proj_ti'] = context_decode_proj_ti

        # 8. Decode sample from activity (task-dependent subspace)
        if 8 in to_do:
            print(8)
            sample_decode_proj_td = self.decode(self.h, self.sample,
                context_pcs, decode_key='sample', 
                save_fn="sample_decode_task_dependent_subspace")
            results['sample_decode_proj_td'] = sample_decode_proj_td

        # 9. Decode sample from activity (task-independent subspace)
        if 9 in to_do:
            print(9)
            sample_decode_proj_ti = self.decode(self.h, self.sample,
                noncontext_pcs, decode_key='sample',
                save_fn="sample_decode_task_independent_subspace")
            results['sample_decode_proj_ti'] = sample_decode_proj_ti

        # 10. Decode context from perturbed activity (raw)
        if 10 in to_do:
            print(10)
            context_decode_pert_raw = self.decode_perturbed(self.h, 
                    self.task_ids,
                    save_fn="context_decode_perturbed_raw")
            results['context_decode_pert_raw'] = context_decode_pert_raw

        # 11. Decode context from perturbed activity (task-dependent subspace)
        if 11 in to_do:
            print(11)
            context_decode_pert_td = self.decode_perturbed(self.h, 
                self.task_ids, context_pcs,
                save_fn="context_decode_perturbed_task_dependent_subspace")
            results['context_decode_pert_td'] = context_decode_pert_td

        # 12. Decode context from perturbed activity (task-independent subspace)
        if 12 in to_do:
            print(12)
            context_decode_pert_ti = self.decode_perturbed(self.h, 
                self.task_ids, noncontext_pcs,
                save_fn="context_decode_perturbed_task_independent_subspace")
            results['context_decode_pert_ti'] = context_decode_proj_ti

        # 13. Cross-temporal decode of context, raw
        if 13 in to_do:
            print(13)
            cross_temp_ctx_raw = self.cross_temporal_decode(self.h,
                self.task_ids, save_fn="cross_temp_context_decode_raw", do_reduce=True)
            results['cross_temp_ctx_raw'] = cross_temp_ctx_raw
         
        # 14. Cross-temporal decode of context, context subspace
        if 14 in to_do:
            print(14)
            cross_temp_ctx_td = self.cross_temporal_decode(self.h, self.task_ids,
                context_pcs, 
                save_fn="cross_temp_context_decode_task_dependent_subspace")
            results['cross_temp_ctx_td'] = cross_temp_ctx_td

        # 15. Cross-temporal decode of context, context-indep. subspace
        if 15 in to_do:
            print(15)
            cross_temp_ctx_ti = self.cross_temporal_decode(self.h, self.task_ids, 
                noncontext_pcs, 
                save_fn="cross_temp_context_decode_task_independent_subspace")
            results['cross_temp_ctx_ti'] = cross_temp_ctx_ti

        # 16. Cross-temporal decode of sample, raw
        if 16 in to_do:
            print(16)
            cross_temp_sam_raw = self.cross_temporal_decode(self.h,
                self.sample, decode_key='sample', 
                save_fn="cross_temp_sample_decode_raw")
            results['cross_temp_sam_raw'] = cross_temp_sam_raw

        # 17. Cross-temporal decode of sample, context subspace
        if 17 in to_do:
            print(17)
            cross_temp_sam_td = self.cross_temporal_decode(self.h, self.sample,
                context_pcs, decode_key='sample',
                save_fn="cross_temp_sample_decode_task_dependent_subspace")
            results['cross_temp_sam_td'] = cross_temp_sam_td

        # 18. Cross-temporal decode of sample, context-indep. subspace
        if 18 in to_do:
            print(18)
            cross_temp_sam_ti = self.cross_temporal_decode(self.h, self.sample, 
                noncontext_pcs, decode_key='sample',
                save_fn="cross_temp_sample_decode_task_independent_subspace")
            results['cross_temp_sam_ti'] = cross_temp_sam_ti

        ########################################################################
        # Epoch-specific analyses
        ########################################################################
        # 19. Identify context subspace using activity from each epoch
        if 19 in to_do:
            print(19)
            context_pcs_by_epoch = self.identify_context_subspace_by_epoch(self.h)

        # 20. Decode context from activity by epoch (task-dependent subspace)
        if 20 in to_do:
            print(20)
            context_decode_ep_proj_td = self.decode_by_epoch(self.h, 
                self.task_ids, context_pcs_by_epoch, 
                save_fn="context_decode_task_dependent_subspace_by_epoch")
            results['context_decode_ep_proj_td'] = context_decode_ep_proj_td

        # 21. Individual unit task selectivity analysis
        if 21 in to_do:
            print(21)
            task_selectivity = self.compute_task_selectivity(self.h)
            results['task_selectivity'] = task_selectivity

        ########################################################################
        # Learning-related analyses
        ########################################################################
        # 22. Compute variance explained of activity for each task
        # when projected onto each task subspace
        if 22 in to_do:
            print(22)
            exp_var_task_sub, accs = [], []
            if hasattr(self, 'train_acts') and self.args.do_learning_analysis:
                train_vars = zip(self.train_acts, self.train_accs)
                for (h_i, acc_i) in train_vars:
                    e_i = self.compute_subspace_projections(h_i, self.train_rules)
                    exp_var_task_sub.append(e_i)
                    accs.append(acc_i)
            else:
                e_end = self.compute_subspace_projections(self.h, self.task_ids)
                exp_var_task_sub.append(e_end)
                accs.append(self.acc)

            exp_var_task_sub = np.array(exp_var_task_sub)
            accs = np.array(accs)
            results['accs'] = accs
            results['exp_var_task_sub'] = exp_var_task_sub

        # 23. Compute separation of activity associated w/ each of the outputs
        # during the test period
        if 23 in to_do:
            act_seps = []
            if hasattr(self, 'train_acts') and self.args.do_learning_analysis:
                train_vars = zip(self.train_acts, self.train_accs)
                for (h_i, acc_i) in train_vars:
                    act_i = self.compute_activity_separation(h_i, self.train_rules, 
                        self.train_labels)
                    act_seps.append(act_i)
            else:
                act_sep_end = self.compute_activity_separation(self.h,
                    self.task_ids, self.labels)
                act_seps.append(act_sep_end)
            act_seps = np.array(act_seps)
            results['act_seps'] = act_seps

        # 24. Compute alignment of decision axis w/ the output readout
        if 24 in to_do:
            alignments = []
            if hasattr(self, 'train_acts') and self.args.do_learning_analysis:
                train_vars = zip(self.train_acts, self.train_accs)
                for i, (h_i, acc_i) in enumerate(train_vars):
                    align_i = self.compute_decision_sep_alignment(h_i, self.train_rules, 
                        self.train_labels, save_fn=f"{i}_points.png")
                    alignments.append(align_i)
            else:
                align_end = self.compute_decision_sep_alignment(self.h,
                    self.task_ids, self.labels)
                alignments.append(align_end)
            alignments = np.array(alignments)
            results['alignments'] = alignments

        ########################################################################
        # Analyses for task-dependence of output
        ########################################################################
        # 25. Compute dimensionality of each task's activity
        task_dimensionality = []

        # 26. Compute separation of activity when projected onto primary axes
        # of decision-related variation for the 
        output_task_dep = []

        # 27. Decode task during test period when projected onto shared axes of 
        # activity separation
        shared_ax_task_decode, shared_ax_separation = [], []

        # 28. Identify response subspace directly
        var_exp_shared_response_subs = []
        out_dec_shared_response_subs = []

        # 29. Time-dependent sequestration analysis
        time_dependent_sequestration = []

        # 30. Compute active signal maintenance by subtracting top-down input
        # directly
        active_task_id_ctx_dec = []
        active_task_id_ctx_cross_temp = []

        # 31. Decode sample stimulus in each task's specific subspace
        sam_per_ctx_dec, sam_per_ctx_cross_temp = [], []

        # 32. Decode output in each task's specific subspace
        out_per_ctx_dec, out_per_ctx_cross_temp = [], []

        # 33. Compute readout angles
        readout_angles = []
        dec_ax_sep     = []
        readout_ax_sep = []

        # 34. Readout alignment analysis
        readout_d_prime = []
        max_d_prime = []

        if hasattr(self, 'train_acts') and self.args.do_learning_analysis:
            train_vars = zip(self.train_acts[-1:], self.train_w_outs[-1:])
            for i, (h_i, w_i) in enumerate(train_vars):

                # 25
                if 25 in to_do:
                    print(25)
                    t_d = self.compute_dimensionality_by_task(h_i, self.train_rules)
                    task_dimensionality.append(t_d)

                # 26
                if 26 in to_do:
                    print(26)
                    o_t_d = self.compute_output_task_dependence(h_i, 
                        self.train_rules, self.train_labels)
                    output_task_dep.append(o_t_d)

                # 27
                if 27 in to_do:
                    print(27)
                    s_a_t_d, s_a_s = \
                        self.compute_joint_output_projection(h_i, self.train_rules, 
                            self.train_labels)
                    shared_ax_task_decode.append(s_a_t_d)
                    shared_ax_separation.append(s_a_s)

                # 28
                if 28 in to_do:
                    print(28)
                    v_e_s_r_s, o_d_s_r_s = self.identify_response_subspace(h_i, 
                        self.train_rules, self.train_labels)
                    var_exp_shared_response_subs.append(v_e_s_r_s)
                    out_dec_shared_response_subs.append(o_d_s_r_s)

                # 29
                if 29 in to_do:
                    print(29)
                    t_d_s = self.compute_time_dependent_sequestration(h_i, 
                        self.train_rules, self.train_labels, n_pcs=task_dimensionality[-1])
                    time_dependent_sequestration.append(t_d_s)

                # 30
                if 30 in to_do:
                    print(30)
                    a_t_i_c_d, a_t_i_c_c_t = \
                        self.compute_active_task_signal_maintenance(h_i,
                            self.train_rules, self.w_td)
                    active_task_id_ctx_dec.append(a_t_i_c_d)
                    active_task_id_ctx_cross_temp.append(a_t_i_c_c_t)

                # 31
                if 31 in to_do:
                    print(31)
                    '''delay_activity = h_i[:,self.epoch_bounds['delay'][-1]-5:self.epoch_bounds['delay'][-1],:]
                    late_delay_dimensionality = self.compute_dimensionality_by_task(delay_activity, 
                        self.train_rules)'''
                    s_p_c_d = []
                    s_p_c_d.append( \
                        self.decode_by_task(h_i,
                            self.train_rules, self.train_samples[:,0], n_pcs=10, do_ct=False))
                    s_p_c_d.append( \
                        self.decode_by_task(h_i,
                            self.train_rules, self.train_samples[:,1], n_pcs=10, do_ct=False))
                    sam_per_ctx_dec.append(s_p_c_d)

                # 32
                if 32 in to_do:
                    print(32)
                    o_p_c_d = \
                        self.decode_by_task(h_i,
                            self.train_rules, self.train_labels,do_ct=False,n_pcs=30)#task_dimensionality[-1][:,4])
                    out_per_ctx_dec.append(o_p_c_d)

                # 33
                if 33 in to_do:
                    print(33)
                    angles, d_s, r_s = self.compute_readout_angle(h_i, 
                        self.train_rules, self.train_labels, w_i)
                    readout_angles.append(angles)
                    dec_ax_sep.append(d_s)
                    readout_ax_sep.append(r_s)

                if 34 in to_do:
                    print(34)
                    r_d_p, m_d_p = self.iterative_readout_alignment_analysis(
                        h_i[:,self.epoch_bounds['test'],:], self.train_rules, self.train_labels,
                        w_i, n_pcs=10)#task_dimensionality[-1][:,4], k=i)
                    readout_d_prime.append(r_d_p)
                    max_d_prime.append(m_d_p)
                
        # 25
        if 25 in to_do:
            task_dimensionality = np.array(task_dimensionality)
            results['task_dimensionality'] = task_dimensionality

        # 26
        if 26 in to_do:
            output_task_dep = np.array(output_task_dep)
            results['output_task_dep'] = output_task_dep

        # 27
        if 27 in to_do:
            shared_ax_task_decode = np.array(shared_ax_task_decode)
            shared_ax_separation = np.array(shared_ax_separation)
            results['shared_ax_task_decode'] = shared_ax_task_decode
            results['shared_ax_separation'] = shared_ax_separation

        # 28
        if 28 in to_do:
            var_exp_shared_response_subs = np.array(var_exp_shared_response_subs)
            out_dec_shared_response_subs = np.array(out_dec_shared_response_subs)
            results['var_exp_shared_response_subs']  = var_exp_shared_response_subs
            results['out_dec_shared_response_subs']  = out_dec_shared_response_subs

        # 29
        if 29 in to_do:
            time_dependent_sequestration = np.array(time_dependent_sequestration)
            results['time_dependent_sequestration']  = time_dependent_sequestration

        # 30
        if 30 in to_do:
            active_task_id_ctx_dec = np.array(active_task_id_ctx_dec)
            active_task_id_ctx_cross_temp = np.array(active_task_id_ctx_cross_temp)
            results['active_task_id_ctx_dec'] = active_task_id_ctx_dec
            results['active_task_id_ctx_cross_temp'] = active_task_id_ctx_cross_temp

        # 31
        if 31 in to_do:
            sam_per_ctx_dec = np.array(sam_per_ctx_dec)
            results['sam_per_ctx_dec'] = sam_per_ctx_dec

        # 32
        if 32 in to_do:
            out_per_ctx_dec = np.array(out_per_ctx_dec)
            results['out_per_ctx_dec'] = out_per_ctx_dec

        # 33
        if 33 in to_do:
            readout_angles = np.array(readout_angles)
            dec_ax_sep = np.array(dec_ax_sep)
            readout_ax_sep = np.array(readout_ax_sep)
            results['readout_angles'] = readout_angles
            results['dec_ax_sep'] = dec_ax_sep
            results['readout_ax_sep'] = readout_ax_sep

        if 34 in to_do:
            results['readout_d_prime'] = readout_d_prime
            results['max_d_prime'] = max_d_prime

        results['accs'] = self.train_accs

        if self.args.do_save:
            pickle.dump(results, open(analysis_save_f, 'wb'))

        # Print the network name
        t = str(datetime.timedelta(seconds=int(round(time.time() - t0, 0))))

        print(self.network_id, t, f"{self.acc.mean():.2f}")


        return results, analysis_save_f


    def cluster_output_weights(self, w_out):
        """
        Perform clustering of network output weights, with number of clusters
        equal to the number of tasks + 1.

        Args:
            w_out (np.ndarray) - output weight matrix, N x n_out
        Returns:
            clustering (list of np.ndarray) - list of vectors of cluster 
                labels, (N,) -- first raw, second normalized
        """

        # 1. Cluster output weights (unnormalized)
        n_clusters = self.N_TASKS + 1
        clustering = KMeans(n_clusters).fit(w_out).labels_

        # 2. Cluster output weights (normalized)
        w_out_norm = w_out / np.amax(w_out, axis=0)
        clustering_norm = KMeans(n_clusters).fit(w_out_norm).labels_

        # If plotting: sort output weight matrix by clustering, plot them, 
        # and save
        if self.args.do_plot:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
            sorted_c = np.argsort(clustering)
            sorted_c_norm = np.argsort(clustering_norm)
            imshow_kwargs = {"interpolation": 'none',
                             "aspect"       : 'auto',
                             "cmap"         : 'viridis'}
            ax[0].imshow(w_out[sorted_c,:], **imshow_kwargs)
            ax[1].imshow(w_out_norm[sorted_c_norm,:], **imshow_kwargs)
            ax[0].set(title='Output clustering (raw)',
                      xlabel='Output unit #',
                      ylabel='Hidden unit #')
            ax[1].set(title='Output clustering (normalized)',
                      xlabel='Output unit #',
                      ylabel='Hidden unit #')
            fig_fn = f"{self.fig_path}_output_projection_clustering.png"
            fig.savefig(fig_fn, dpi=300)
            plt.close(fig)

        return [clustering, clustering_norm]

    def compute_output_projection(self, h, rules, w_out, k_folds=4):
        """
        Classify which task is being performed from the relationship b/w
        activity during that task + the readout axis.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rules (B x 1)
            w_out (np.ndarray) - output weight matrix, N x n_out
        Returns:
            task_decode (np.ndarray) - decoding of task @ each timepoint

        """
         # Subset activity to focus on test period
        h_tp = h[:,self.epoch_bounds['test'],:]
        h_tp = h_tp[:,2:,:]
        w_out_dec_axis = PCA(1).fit(self.w_out[:,1:].T).components_

        output_proj = np.zeros_like(h_tp)

        # Compute output projections
        for i in range(h_tp.shape[0]):
            h_rel = np.reshape(h_tp[i], (-1, self.N_NEUR))
            had_prods = np.array([np.multiply(p, w_out_dec_axis) for p in h_rel])
            output_proj[i] = np.reshape(had_prods, h_tp[i].shape)

        # Decode, timepoint by timepoint, which task is being performed
        task_decode = self.decode(output_proj, rules)

        return task_decode

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

    def identify_context_subspace_by_epoch(self, h):
        """
        Perform Murray-type analysis to identtify subspace, epoch-
        by-epoch, where context is consistently discriminable; when 
        plotting, project activity from other epochs into these subspaces.
        """
        context_by_epoch = {}
        for j, (k, b) in enumerate(self.epoch_bounds.items()):
            # Supply activity from this epoch to identify context subspace
            cur_h = h[:,b,:]
            context_by_epoch[k] = self.identify_context_subspace(cur_h, 
                do_plot=False)
        task_tr = [np.where(self.task_ids == i)[0] for i in 
            np.unique(self.task_ids)]

        # If plotting: project activity from each 
        if self.args.do_plot:
            colors, _, _ = self.get_colors()
            fig, ax = plt.subplots(nrows=len(context_by_epoch),
                ncols=len(context_by_epoch), figsize=(10,10))
            for i, (k1, v1) in enumerate(context_by_epoch.items()):
                for j, (k2, v2) in enumerate(context_by_epoch.items()):
                    raw_h = np.reshape(h[:,self.epoch_bounds[k1],:],
                        (-1, self.N_NEUR))
                    trans_h = v2.transform(raw_h)[:,:2]
                    trans_h = np.reshape(trans_h,
                        (h.shape[0], len(self.epoch_bounds[k1]), -1))
                    for k, t in enumerate(np.unique(self.task_ids)):
                        h_to_plot = trans_h[task_tr[k],...]
                        for trial in h_to_plot:
                            ax[i,j].plot(trial[:,0], trial[:,1], color=colors[k])
                    ax[i,j].set(title=f"{k1} onto {k2}",
                        xlabel="PC1", ylabel="PC2")

            # Add titles/labels etc
            fig_fn = f"{self.fig_path}_task_subspace_projections_by_epoch.png"
            plt.tight_layout()
            fig.savefig(fig_fn, dpi=300)
            plt.close(fig)

        return context_by_epoch

    def identify_context_independent_subspace(self, h, context_pcs):
        """
        Perform Murray-type analysis to identify task-independent subspace
        (PCA with T input samples, one per timepoint, consisting of average 
        activity vector across all trials/tasks at that timepoint).

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
        Returns:
            noncontext_pca (sklearn PCA() object) - basis of task-ind. subspace
            var_exp (np.ndarray) - cumulative variance explained for comp. 1-N
        """

        # Remove components of activity in context-dependent subspace
        # First: project all points into the subspace
        h_cd_subs = np.reshape(h, (-1, self.N_NEUR))
        h_cd_subs = context_pcs.inverse_transform(
            context_pcs.transform(h_cd_subs))
        h_cd_subs = np.reshape(h_cd_subs, (h.shape[0], h.shape[1], -1))
        h = h - h_cd_subs

        # Average activity across all trials w/ same sample stimulus
        h_by_sample = [h[np.where(self.sample == i)[0],:,:] for \
            i in np.unique(self.sample)]
        late_delay = range(self.epoch_bounds['delay'][-1] - 10, 
            self.epoch_bounds['delay'][-1])
        mean_time_h = np.array([np.mean(h_i[:,late_delay,:], axis=(0,1)) 
            for h_i in h_by_sample])

        # Perform PCA on time*tasks x N matrix to identify ctx-ind. subspace
        noncontext_pca = PCA(self.args.N_PCS).fit(mean_time_h)

        # Quantify amount of variance explained by these ctx-ind. components
        all_data = np.reshape(h, (-1, self.N_NEUR))
        var_exp = np.zeros(self.args.N_PCS)
        for i in range(self.args.N_PCS):
            pca_i = PCA(i + 1).fit(mean_time_h)
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
        if self.args.do_plot:
            task_tr = [np.where(self.task_ids == i)[0] for i in 
                np.unique(self.task_ids)]
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
                h_trans = noncontext_pca.transform(h_orig)[:,:2]
                h_trans = np.reshape(h_trans, 
                    (h[task_tr[i]].shape[0], h[task_tr[i]].shape[1], 2))
                for trial in h_trans:
                    ax[0].plot(trial[:,0], trial[:,1], color=colors[i])

                # (b) Plot of mean across timepoints per epoch, for all trials
                for j, (k,b) in enumerate(self.epoch_bounds.items()):
                    h_orig = h[task_tr[i],...]
                    h_orig = h_orig[:,b,:].mean(1)
                    h_trans = noncontext_pca.transform(h_orig)[:,:2]
                    ax[j + 1].scatter(h_trans[:,0], h_trans[:,1], color=colors[i])

            # Add titles/labels etc
            ax[0].set(title="All trials/timepoints", xlabel="PC1", ylabel="PC2")
            for j, (k, v) in enumerate(self.epoch_bounds.items()):
                ax[j + 1].set(title=k, xlabel="PC1", ylabel="PC2")
            plt.tight_layout()
            fig_fn = f"{self.fig_path}_task_indep_subspace_projections.png"
            fig.savefig(fig_fn, dpi=300)
            plt.close(fig)

        return noncontext_pca, var_exp

    def identify_pcs_per_task(self, h):
        """
        Perform PCA on activity for each task separately; return the components.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
        Returns:
            per_task_pcs (list of np.ndarray, K x N) - top K PCs for each task
            vars_explained (np.ndarray, N_TASKS x K)
        """
        task_tr = [np.where(self.task_ids == i)[0] for i in 
            np.unique(self.task_ids)]
        all_pcas = []
        per_task_pcs = []
        vars_explained = np.zeros((self.N_TASKS, self.args.N_PCS))

        # For each task: identify task axes
        h = h[:,self.epoch_bounds['delay'][0]+5:self.epoch_bounds['delay'][-1],:]
        for i, task in enumerate(np.unique(self.task_ids)):
            h_to_fit = np.reshape(h[task_tr[i]], (-1, self.N_NEUR))
            task_pca = PCA(self.args.N_PCS).fit(h_to_fit)
            all_pcas.append(task_pca)
            per_task_pcs.append(task_pca)
            vars_explained[i,:] = task_pca.explained_variance_ratio_

        # If plotting: for each task, project activity onto appropriate
        # components (no averaging across trials) and plot
        if self.args.do_plot:

            colors, _, _ = self.get_colors()

            fig, ax = plt.subplots(nrows=self.N_TASKS, ncols=1, figsize=(5,10))
            for i, task in enumerate(np.unique(self.task_ids)):
                h_raw = np.reshape(h[task_tr[i]], (-1, self.N_NEUR))
                h_trans = all_pcas[i].transform(h_raw)
                h_trans = np.reshape(h_trans, (h[task_tr[i]].shape[0], 
                    h[task_tr[i]].shape[1], -1))
                for trial in h_trans:
                    ax[i].plot(trial[:,0], trial[:,1], color=colors[i])
                ax[i].set(title=f"Task {i}", xlabel="PC1", ylabel="PC2")
            plt.tight_layout()
            fig_fn = f"{self.fig_path}_per_task_pca.png"
            fig.savefig(fig_fn, dpi=300)
            plt.close(fig)

        return per_task_pcs, vars_explained

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
        # Generate k-folds
        skf = StratifiedKFold(n_splits=self.args.k_folds)
        scores = np.zeros((h.shape[1], self.args.k_folds))
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

        # For each fold: at each timepoint, decode and store
        for i, (tr_idx, te_idx) in enumerate(skf.split(h[:, 0, :], y)):
            y_tr, y_te = y[tr_idx], y[te_idx]
            for t in range(h.shape[1]):
                X_tr, X_te = h[tr_idx, t, :], h[te_idx, t,  :]
                ss = StandardScaler()
                ss.fit(X_tr)
                X_tr = ss.transform(X_tr)
                X_te = ss.transform(X_te)
                svm.fit(X_tr, y_tr)
                scores[t, i] = svm.score(X_te, y_te)
        
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

        return scores.mean(1)

    def decode_perturbed(self, h, y,
        context_comp=None, decode_key='context', save_fn=None):
        """
        Decode labels from activity, but with some perturbations (randomly 
        excluding or zeroing out an increasing fraction of units for each test,
        to determine whether decoding is supported by few neurons or is 
        strongly distributed).

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            y (np.ndarray) - labels (B x 1)
            context_comp (np.ndarray) - context subspace to project into
        Returns:
            context_decode (np.ndarray) - decoding of context @ each timepoint
        """
        # For the perturbation fractions: sparsely sample the low perturbations,
        # densely sample the high perturbations
        perturbation_fractions = np.concatenate(
            (np.linspace(0, 0.95, self.args.n_perturbation_fracs),
             1-np.linspace(0., 0.05, self.args.n_perturbation_fracs)**2))
        perturbation_fractions.sort()
        scores = np.zeros((len(perturbation_fractions), self.args.n_repetitions,
            h.shape[1]))

        for i, f in enumerate(perturbation_fractions):
            for j in range(self.args.n_repetitions):
                # Randomly choose f pct of neurons to 0 out
                to_zero = np.random.choice(self.N_NEUR, int(f * self.N_NEUR), 
                    replace=False)
                h_copy = h.copy()
                h_copy[:,:,to_zero] = np.random.permutation(h_copy[:,:,to_zero])
                scores[i,j,:] = self.decode(h_copy, y,
                     context_comp, do_plot=False)

        scores = scores.mean(1)

        if self.args.do_plot:

            colors = plt.cm.inferno(np.linspace(0., 1., 
                self.args.n_perturbation_fracs*2))

            fig, ax = plt.subplots(1)
            for i in range(scores.shape[0]):
                ax.plot(scores[i], color=colors[i], 
                    label=f"{perturbation_fractions[i]:.4f}")

            ax.hlines(y=1./len(np.unique(y)), xmin=0, xmax=scores.shape[1] - 1, 
                color='black', linestyle='--', label='Chance')
            ax.legend()
            ax.set(title=f"Perturbed {decode_key} decode",
                xlabel="Timestep", ylabel="Accuracy",
                ylim=[0.,1.])
            plt.tight_layout()

            # Build filename 
            if save_fn is None:
                fig_fn = f"{self.fig_path}_context_decode_perturbed.png"
            else:
                fig_fn = f"{self.fig_path}_{save_fn}.png"
            fig.savefig(fig_fn, dpi=300)
            plt.close(fig)

        return scores

    def decode_by_epoch(self, h, y, 
        context_comp_by_ep=None, save_fn=None, do_plot=True, raw_n_comp=50):
        """
        Decode task ID from activity by epoch.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            y (np.ndarray) - labels (B x 1)
            context_comp_by_ep (sklearn PCA() obj.) - context subsp. to 
                project into
        Returns:
            scores (dict of np.ndarray) - decoding of context @ each timepoint 
                in each epoch; key = (subspace, decode) epoch tuple, where 
                subspace gives the epoch from which context PCs were obtained
                and decode gives the epoch whose activity is projected onto 
                those components and from which task identity is decoded
        """
        # Generate k-folds
        skf = StratifiedKFold(n_splits=self.args.k_folds)
        all_scores = {}
        svm = SVC(C=1.0, kernel='linear', max_iter=1000,decision_function_shape="ovr",shrinking=False, tol=1e-3)
        
        for sub_i, (k_sub, v_sub) in enumerate(self.epoch_bounds.items()):

            # Decode from all other epochs' activity projected into this subspace
            for dec_i, (k_dec, v_dec) in enumerate(self.epoch_bounds.items()):

                h_sub = h[:,v_sub,:]
                h_dec = h[:,v_dec,:]
                scores = np.zeros((len(v_dec), self.args.k_folds))

                # Set up subspace to project into
                # If any components specified, project onto them before decode
                if context_comp_by_ep[k_sub] is not None:
                    h_to_project = np.reshape(h_dec, (-1, self.N_NEUR))
                    h_projected = context_comp_by_ep[k_sub].transform(h_to_project)
                    h_dec = np.reshape(h_projected, 
                        (h_dec.shape[0], h_dec.shape[1], -1))

                # If not specified, reduce in dimensionality to e.g. 200D
                else:
                    h_to_project = np.reshape(h_dec, (-1, self.N_NEUR))
                    h_to_fit = np.reshape(h_sub, (-1, self.N_NEUR))
                    pca = PCA(raw_n_comp).fit(h_to_fit)
                    h_projected = pca.transform(h_to_project)
                    h_dec = np.reshape(h_projected, 
                        (h_dec.shape[0], h_dec.shape[1], -1))
            
                # For each fold: at each timepoint, decode and store
                for i, (tr, te) in enumerate(skf.split(h_dec[:, 0, :], y)):
                    y_tr, y_te = y[tr], y[te]
                    for t in range(h_dec.shape[1]):
                        X_tr, X_te = h_dec[tr, t, :], h_dec[te, t,  :]
                        ss = StandardScaler()
                        ss.fit(X_tr)
                        X_tr = ss.transform(X_tr)
                        X_te = ss.transform(X_te)
                        svm.fit(X_tr, y_tr)
                        scores[t, i] = svm.score(X_te, y_te)

                # Save decode for this subspace/decode epoch combo
                all_scores[(k_sub, k_dec, sub_i, dec_i)] = scores.mean(1)
        
        # If plotting: plot the decode through time (mean across folds)
        if self.args.do_plot and do_plot:

            fig, ax = plt.subplots(nrows=len(self.epoch_bounds.items()), 
                ncols=len(self.epoch_bounds.items()), figsize=(12,12))
            for key, val in all_scores.items():
                row, col = key[-2:]
                ax[row, col].plot(val, label='Decode')
                ax[row, col].hlines(y=1./len(np.unique(y)), xmin=0, 
                    xmax=val.shape[0] - 1, color='black', linestyle='--', 
                    label='Chance')
                ax[row, col].set(title=f"{key[0]} sbsp., {key[1]} act.", 
                    xlabel="Timestep", ylabel="Accuracy")
            plt.tight_layout()

            # Build filename 
            if save_fn is None:
                fig_fn = f"{self.fig_path}_cross_epoch_context_decode.png"
            else:
                fig_fn = f"{self.fig_path}_{save_fn}.png"
            fig.savefig(fig_fn, dpi=300)
            plt.close(fig)

        return scores.mean(1)

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
        # Generate k-folds
        skf = StratifiedKFold(n_splits=self.args.k_folds)
        scores = np.zeros((int(np.ceil(h.shape[1]/2)), int(np.ceil(h.shape[1]/2)), self.args.k_folds))
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
        for i, (tr_idx, te_idx) in enumerate(skf.split(h[:, 0, :], y)):
            y_tr, y_te = y[tr_idx], y[te_idx]
            for t_tr in range(0,h.shape[1],2):
                for t_te in range(t_tr, h.shape[1],2):
                    X_tr = h[tr_idx, t_tr, :]
                    X_te = h[te_idx, t_te,  :]
                    ss = StandardScaler()
                    ss.fit(X_tr)
                    X_tr = ss.transform(X_tr)
                    X_te = ss.transform(X_te)
                    svm.fit(X_tr, y_tr)
                    scores[t_tr//2, t_te//2, i] = svm.score(X_te, y_te)
        
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

    def compute_task_selectivity(self, h, do_plot=True,
        save_fn=None):
        """
        Compute individual units' selectivity for single tasks.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
        Returns:
            sel (np.ndarray, (N,)) - task selectivity indices for each neuron

        """
        # For each neuron: compute difference in activity across different 
        # trials of the same task vs. different trials of different tasks
        # (that is: timepoint by timepoint, compute the mean distance between
        # all same-task trials at that t vs. all different-task trials at
        # that t)

        # Compute within vs. across task distances for each
        # task, to determine maximal task selectivity index
        WTD = np.zeros((self.N_NEUR, h.shape[1], self.N_TASKS, self.N_SAMPLES))
        BTD = np.zeros((self.N_NEUR, h.shape[1], self.N_TASKS, self.N_SAMPLES))
    
        for i,t in enumerate(np.unique(self.task_ids)):
            for j,s in enumerate(np.unique(self.sample)):
                for timepoint in range(h.shape[1]):
                    cur_task_h = h[np.where((self.sample == s) & 
                        (self.task_ids == t))[0],timepoint,...].squeeze()
                    other_tasks_h = h[np.where((self.sample == s) & 
                        (self.task_ids != t))[0],timepoint,...].squeeze()
                    wtd = np.std(cur_task_h, axis=0)
                    btd = np.std(other_tasks_h, axis=0)
                    WTD[:,timepoint,i,j] = wtd
                    BTD[:,timepoint,i,j] = btd

        # Compute selectivity index for each unit at each timestep
        BTD_t = np.nanmean(BTD, axis=3)
        WTD_t = np.nanmean(WTD, axis=3)
        numerator_t = np.amax(BTD_t, axis=2) - np.amax(WTD_t, axis=2)
        denominator_t = np.amax(BTD_t, axis=2) + np.amax(WTD_t, axis=2)
        task_selectivity_t = np.divide(numerator_t, denominator_t)

        WTD = np.nanmean(WTD, axis=(1,3))
        BTD = np.nanmean(BTD, axis=(1,3))

        # Compute selectivity index for each unit in aggregate
        numerator = np.amax(BTD, axis=1) - np.amax(WTD, axis=1)
        denominator = np.amax(BTD, axis=1) + np.amax(WTD, axis=1)
        task_selectivity = np.divide(numerator, denominator)

        if self.args.do_plot and do_plot:
            fig = plt.figure(figsize=(10,5))
            ax0 = fig.add_subplot(121)
            ax1 = fig.add_subplot(122, projection='3d')
            ax = [ax0, ax1]
            bins = np.linspace(-1., 1., 100)

            # Average histogram of task selectivity indices
            ax[0].hist(task_selectivity, bins=bins)
            ax[0].axvline(0, color='black', linestyle='--')
            ax[0].set(ylabel="Count", xlabel="TSI", 
                title="Task selectivity distribution",
                xlim=[-1., 1.])

            # Time-dependent histogram
            hists = np.array([np.histogram(ts_t, bins=bins)[0] for ts_t in 
                task_selectivity_t.T]).T

            x = np.arange(0, hists.shape[1])
            y = np.linspace(-1, 1, hists.shape[0])
            X, Y = np.meshgrid(x, y)
            ax[1].plot_surface(Y, X, hists, cmap=plt.cm.plasma)
            ax[1].view_init(20, 285)
            ax[1].set_xticks(np.linspace(-1, 1., 5))
            ax[1].set_xticklabels([f"{l:.2f}" for l in np.linspace(-1, 1, 5)])
            ax[1].set(title="Task selectivity vs. time",
                xlabel="TSI", ylabel="Timestep", zlabel="Count (# units)")

            plt.tight_layout()
            # Build filename 
            if save_fn is None:
                fig_fn = f"{self.fig_path}_task_selectivity_indices.png"
            else:
                fig_fn = f"{self.fig_path}_{save_fn}.png"
            fig.savefig(fig_fn, dpi=300)
            plt.close(fig)

        return

    def compute_subspace_projections(self, h, rules,
        do_plot=True, save_fn=None):
        """
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

        # Perform PCA on each task activity separately, then project
        # other tasks' activities onto these axes and compute var. exp. 
        # (here, measured as R2 between original activity + reconstruction
        # from low-D projection)
        var_ex = np.zeros((self.N_TASKS, 2))
        h_by_task = {j: h[np.where(rules == j)[0]] for j in np.unique(rules)}
        h_means = {k: v.mean(0) for k, v in h_by_task.items()}
        h_by_task = {k: np.reshape(v, (-1, self.N_NEUR)) 
            for k, v in h_by_task.items()}
        rule_options = np.unique(rules)
        for i, t1 in enumerate(rule_options):
            pca_i = PCA(self.args.N_MAX_PCS).fit(h_means[t1])
            data_non_t1 = np.vstack([h_by_task[t2] 
                for t2 in np.setdiff1d(rule_options, t1)])
            h_i_subspace_i = pca_i.transform(h_by_task[t1])
            h_j_subspace_i = pca_i.transform(data_non_t1)
            var_ex[i,0] = r2_score(h_by_task[t1], 
                pca_i.inverse_transform(h_i_subspace_i),
                multioutput='variance_weighted') 
            var_ex[i,1] = r2_score(data_non_t1, 
                pca_i.inverse_transform(h_j_subspace_i),
                multioutput='variance_weighted')     

        if self.args.do_plot and do_plot:
            pass

        return var_ex

    def compute_activity_separation(self, h, rules, outputs, do_plot=True, 
        save_fn=None):
        """
        Compute the separation in activity that corresponds w/ each output
        type for each task, separately. Report as a ratio -- mean distance 
        between h on pairs of trials w/ same required output / mean distance
        between h on pairs of trials w/ opposite required output.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            outputs (np.ndarray) - req. decision identifier for batch (B x 1)
        Returns:
            act_sep (np.ndarray, (N_TASKS x 1)) -- separation index of test-
                period activity across decisions, task by task
    
        """
        h_tp = h[:,self.epoch_bounds['test'],:]
        act_sep = np.zeros(self.N_TASKS)
        for i, t in enumerate(np.unique(rules)):
            # Select activity corresponding w/ each task/output
            rel_h_o1 = h_tp[np.where((rules == t) & (outputs == 1))[0]]
            rel_h_o2 = h_tp[np.where((rules == t) & (outputs == 2))[0]]

            # Consolidate all activity vectors w/ same desired output
            rel_h_o1 = np.reshape(rel_h_o1, (-1, self.N_NEUR))
            rel_h_o2 = np.reshape(rel_h_o2, (-1, self.N_NEUR))

            # Compute pairwise distances within each bundle
            mean_dist_o1 = pdist(rel_h_o1, 'cityblock').mean()
            mean_dist_o2 = pdist(rel_h_o2, 'cityblock').mean()
            mean_dist_o1_o2 = cdist(rel_h_o1, rel_h_o2, 'cityblock').mean()

            # Compute index of separation for this task (mean distance 
            # among activity traces on trials needing same output divided
            # by mean distance between activities on trials needing 
            # different outputs)
            numerator = np.mean([mean_dist_o1, mean_dist_o2])
            act_sep[i] = numerator / mean_dist_o1_o2

        return act_sep

    def compute_decision_sep_alignment(self, h, rules, outputs, subspace=None,
        do_plot=True, save_fn=None):
        """ 
        Use the Murray approach to identify decision axis from h;
        also use Murray approach to identify decision axis from output weights;
        measure alignment of output vector w/ separation in h as d-prime of
        projection of test h onto decision axis defined from outputs; measure
        separation of h directly via projection onto the h-def. decision axis.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            outputs (np.ndarray) - req. decision identifier for batch (B x 1)
            subspace (sklearn PCA() obj.) - subspace to project into
        Returns:
            alignments (np.ndarray) - task alignment scores (N_TASKS x 2);
                column 0 = d-prime of activity projected onto readout axis,
                column 1 = d-prime of activity projected onto activity-sep axis
        """

        # Subset activity to focus on test period
        h_tp = h[:,self.epoch_bounds['test'],:]
        h_tp = h_tp[:,2:,:]
        alignments = np.zeros((self.N_TASKS,2))
        w_out_dec_axis = PCA(1).fit(self.w_out[:,1:].T)

        # Project h_tp into subspace if specified
        if subspace is not None:
            h_tp_all = subspace.transform(np.reshape(h_tp, (-1, self.N_NEUR)))
            h_tp = np.reshape(h_tp_all, (h_tp.shape[0], h_tp.shape[1], -1))
            w_out_reduced = subspace.transform(self.w_out[:,1:].T)
            w_out_dec_axis = PCA(1).fit(w_out_reduced)

        for i, t in enumerate(np.unique(rules)):
            # Select activity corresponding w/ each task/output
            rel_h_o1 = h_tp[np.where((rules == t) & (outputs == 1))[0]]
            rel_h_o2 = h_tp[np.where((rules == t) & (outputs == 2))[0]]

            # Take all test-period activity and collect into 2d matrix
            # for PCA fit
            # OLD: fit PCA from means of activity corresponding w/ different 
            #   outputs
            all_data = np.vstack(
                 (rel_h_o1.mean((0,1)), rel_h_o2.mean((0,1))))
            # NEW: fit PCA using all data (just to see what separation exists,
            #   whether or not it is correct/productive)
            #all_data = np.reshape(h_tp, (-1, h_tp.shape[-1]))

            # Fit axis along which cross-output variation exists
            act_dec_axis = PCA(1).fit(all_data)

            # Project onto w_out_dec_axis and act_dec_axis
            h_w_dec_o1 = w_out_dec_axis.transform(
                np.reshape(rel_h_o1, (-1, rel_h_o1.shape[-1])))
            h_w_dec_o2 = w_out_dec_axis.transform(
                np.reshape(rel_h_o2, (-1, rel_h_o2.shape[-1])))
            h_act_dec_o1 = act_dec_axis.transform(
                np.reshape(rel_h_o1, (-1, rel_h_o1.shape[-1])))
            h_act_dec_o2 = act_dec_axis.transform(
                np.reshape(rel_h_o2, (-1, rel_h_o2.shape[-1])))

            # Compute amount of separation between output groups for both axes
            dp_w = d_prime(h_w_dec_o1, h_w_dec_o2)
            dp_h = d_prime(h_act_dec_o1, h_act_dec_o2)

            # Record these results
            alignments[i,:] = [dp_w, dp_h]

        return alignments

    def compute_output_task_dependence(self, h, rules, outputs, do_plot=True,
        save_fn=None):
        """
        Similar to sequestration analysis, but focused on test period: find
        the axis of separation between o1 vs. o2 for each task; project activity
        from other tasks onto this axis, and determine how much separability 
        exists w.r.t. output.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            outputs (np.ndarray) - req. decision identifier for batch (B x 1)
        Returns:
            separation (np.ndarray) - decode of output when projected onto 
                each task's axes of output-separation (N_TASKS x 2)
        """
        separation = np.zeros((self.N_TASKS, 2))

        # Subset activity to focus on test period
        h_tp = h[:,self.epoch_bounds['test'],:]
        h_tp = h_tp[:,2:,:] # To skip training mask
        w_out_dec_axis = PCA(2).fit(self.w_out[:,1:].T)

        # For each task: find axis of separation
        for i, t1 in enumerate(np.unique(rules)):
            # Select activity corresponding w/ each task/output
            rel_h_o1_i = h_tp[np.where((rules == t1) & (outputs == 1))[0]]
            rel_h_o2_i = h_tp[np.where((rules == t1) & (outputs == 2))[0]]
            all_data_i = np.vstack(
                 (rel_h_o1_i.mean(0), rel_h_o2_i.mean(0)))
            pca_i = PCA(5).fit(all_data_i)

            # Select activity corresponding w/ other tasks
            rel_h_o1_j = h_tp[np.where((rules != t1) & (outputs == 1))[0]]
            rel_h_o2_j = h_tp[np.where((rules != t1) & (outputs == 2))[0]]

            # Reshape
            rel_h_o1_j = np.reshape(rel_h_o1_j, (-1, self.N_NEUR))
            rel_h_o2_j = np.reshape(rel_h_o2_j, (-1, self.N_NEUR))
            rel_h_o1_i = np.reshape(rel_h_o1_i, (-1, self.N_NEUR))
            rel_h_o2_i = np.reshape(rel_h_o2_i, (-1, self.N_NEUR))

            separation[i,0] = decode_output(pca_i.transform(rel_h_o1_i),
                pca_i.transform(rel_h_o2_i)).mean(-1)

            separation[i,1] = decode_output(pca_i.transform(rel_h_o1_j),
                pca_i.transform(rel_h_o2_j)).mean(-1)

        return separation

    def compute_joint_output_projection(self, h, rules, outputs, k_folds=4, 
        do_plot=True, save_fn=None):
        """
        Pools all activity across tasks on trials w/ same output to find 
        "task independent" axes of variation btwn outputs during test period;
        project onto these axes and try to:
        (a) decode task ID, 
        (b) compute separability in activity just projected onto these axes
        (c) compute separability in activity projected onto output dimension

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            outputs (np.ndarray) - req. decision identifier for batch (B x 1)
        Returns:
            task_decode (np.ndarray) - decoding of task from projection onto 
                common separation axis (T x k_folds)
            separation (np.ndarray) - separation scores (2 x 1);
                column 0 = d-prime of activity projected onto readout axis,
                column 1 = d-prime of activity projected onto activity-sep axis
        """
        separation  = []

        # Subset activity to focus on test period
        h_tp = h[:,self.epoch_bounds['test'],:]
        h_tp = h_tp[:,2:,:] # To skip training mask
        h_tp = np.maximum(0, h_tp)
        w_out_dec_axis = PCA(2).fit(self.w_out[:,1:].T)

        # Average activity based on required output and identify axes of var.
        # First: do this by task, then average across tasks to ensure equal
        # representation for all of the tasks + no biasing of axes 
        h_o1_means, h_o2_means = [], []
        for t in np.unique(rules):
            rel_t_o1 = h_tp[np.where((outputs == 1) & (rules == t))[0]]
            rel_t_o2 = h_tp[np.where((outputs == 2) & (rules == t))[0]]
            h_o1_means.append(rel_t_o1.mean((0,1)))
            h_o2_means.append(rel_t_o2.mean((0,1)))
        h_means = np.vstack((np.array(h_o1_means).mean(0), 
            np.array(h_o2_means).mean(0)))
        h_o1 = h_tp[np.where(outputs == 1)[0]]
        h_o2 = h_tp[np.where(outputs == 2)[0]]
        shared_sep_axes = PCA(2).fit(h_means)

        # Project activity onto shared axes
        h_o1_proj = shared_sep_axes.transform(
            np.reshape(h_o1, (-1, self.N_NEUR)))
        h_o2_proj = shared_sep_axes.transform(
            np.reshape(h_o2, (-1, self.N_NEUR)))

        # Project also onto output axes
        h_o1_rec = w_out_dec_axis.transform(
            shared_sep_axes.inverse_transform(h_o1_proj))
        h_o2_rec = w_out_dec_axis.transform(
            shared_sep_axes.inverse_transform(h_o2_proj))

        # Decode task ID from projection onto shared axes
        all_proj = shared_sep_axes.transform(
            np.reshape(h_tp, (-1, self.N_NEUR)))
        all_proj = np.reshape(all_proj, (h_tp.shape[0], h_tp.shape[1], -1))
        task_decode = self.decode(all_proj, rules, return_all=True)

        # Compute separation scores (projected into shared subspace,
        # projected from shared subspace onto output)
        all_proj_out = w_out_dec_axis.transform(
            shared_sep_axes.inverse_transform(
                shared_sep_axes.transform(np.reshape(h_tp, (-1, self.N_NEUR)))))
        all_proj_out = np.reshape(all_proj_out, 
            (h_tp.shape[0], h_tp.shape[1], -1))
        separation.append(self.decode(all_proj, outputs))
        separation.append(self.decode(all_proj_out, outputs))
        separation = np.array(separation)


        return task_decode, separation

    def identify_response_subspace(self, h, rules, outputs, 
        do_plot=True, save_fn=None):
        """
        Look directly for motion of activity into shared subspace during 
        test period (or at least for output-potent subspace that is shared 
        among tasks); start by subtracting projection onto axes of variation 
        identified during sample/delay, then identify axes of variation among
        all trials averaged at each timepoint; this should isolate variation
        that is test-period-specific, and is a good place to start looking.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            outputs (np.ndarray) - req. decision identifier for batch (B x 1)
        Returns:
            var_exp (np.ndarray) - variance explained per shared component 
                (N_PCS x 1)
        """
        # Separate test-period vs. all other activity
        h_tp = h[:,self.epoch_bounds['test'],:]
        h_tp = h_tp[:,2:,:]

        h_tp = np.maximum(0, h_tp)

        # Identify principal axes of variation in the data
        # across the two outputs
        h_tp_o1  = h_tp[np.where(outputs == 1)[0]].mean(0)
        h_tp_o2  = h_tp[np.where(outputs == 2)[0]].mean(0)
        to_fit   = np.concatenate((h_tp_o1, h_tp_o2), axis=0)
        all_data = np.reshape(h_tp, (-1, self.N_NEUR))

        pca = PCA(10).fit(to_fit)
        h_tp_recon = pca.inverse_transform(
            pca.transform(all_data))
        var_exp = r2_score(all_data, 
                           h_tp_recon,
                           multioutput='variance_weighted') 

        # Compute accuracy of classification when projected into resp. sub.
        all_data_reshaped = np.reshape(pca.transform(all_data), 
            (h_tp.shape[0], h_tp.shape[1], -1))
        separation = self.decode(all_data_reshaped, outputs)

        return var_exp, separation

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
        """
        # If only subset of data provided, don't try to execute for all epochs
        
        if h.shape[1] < self.epoch_bounds['test'][-1]:
            # Participation ratio: fit PCA, take percent explained
            dim_part_rat = np.zeros(self.N_TASKS)
            
            for i, t in enumerate(np.unique(rules)):
                h_t = h[np.where(rules == t)[0]]
                pca = PCA().fit(np.reshape(h_t, (-1, self.N_NEUR)))
                e_vals = pca.explained_variance_ratio_
                dim_part_rat[i] = np.sum(e_vals)**2 / np.sum(e_vals**2)

        else:
            # Participation ratio: fit PCA, take percent explained
            dim_part_rat = np.zeros((self.N_TASKS, len(self.epoch_bounds)))
            
            for i, t in enumerate(np.unique(rules)):
                for j, b in enumerate(self.epoch_bounds.values()):
                    h_t = h[np.where(rules == t)[0]]
                    pca = PCA().fit(np.reshape(h_t[:,b,:], (-1, self.N_NEUR)))
                    e_vals = pca.explained_variance_ratio_
                    dim_part_rat[i,j] = np.sum(e_vals)**2 / np.sum(e_vals**2)

        return dim_part_rat

    def compute_time_dependent_sequestration(self, h, rules, outputs, n_pcs=args.N_PCS,
        do_plot=True):
        """ 
        Time-dependent sequestration analysis (look @ principal axes for 
        each task defined at windows of 10 time steps, or during each epoch);
        see if, across training, the time-dependent sequestration changes 
        (maybe the sequestration needs to specifically decrease during the test
        period, which current analysis cannot pick up). 

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            outputs (np.ndarray) - req. decision identifier for batch (B x 1)
        Returns: 
            var_ex (np.ndarray) -- variance explained, 
                (T/step x N_TASKS x N_TASKS); variance explained for projecting
                activity during task j onto principal axes of variation for task
                i at timepoint k (element (i, j, k))
        """

        # For each group of window timesteps, perform PCA on each task activity
        # separately; then project other tasks' activities onto these axes 
        # and compute var. exp. (here, measured as R2 between original activity
        # + reconstruction from low-D projection)
        starts = [self.epoch_bounds['dead'][0],
                  self.epoch_bounds['fix'][0], 
                  self.epoch_bounds['sample'][0], 
                  self.epoch_bounds['delay'][0], 
                  self.epoch_bounds['test'][0]]
        ends   = [self.epoch_bounds['dead'][1],
                  self.epoch_bounds['fix'][1], 
                  self.epoch_bounds['sample'][1], 
                  self.epoch_bounds['delay'][1], 
                  self.epoch_bounds['test'][1]]
        h_by_task = {j: h[np.where(rules == j)[0]] for j in np.unique(rules)}
        h_by_task = [{k: np.reshape(v[:,s:e,:], (-1, self.N_NEUR)) 
                for k, v in h_by_task.items()} for s,e in zip(starts, ends)]
        var_ex = []
        if type(n_pcs) is int:
            n_pcs = np.repeat(n_pcs, (len(starts), self.N_TASKS))
        for k, (s, e) in enumerate(zip(starts, ends)):
            v_e = np.zeros((self.N_TASKS, 2))
            h_by_task_k = h_by_task[k]
            rule_options = np.unique(rules)
            for i, t1 in enumerate(rule_options):
                pca_i = PCA(int(n_pcs[i, k])+1).fit(h_by_task_k[t1])
                h_by_task_t2s = np.vstack([h_by_task_k[t2] for t2 in 
                    np.setdiff1d(rule_options, t1)])
                h_j_subspace_i = pca_i.transform(h_by_task_t2s)
                h_i_subspace_i = pca_i.transform(h_by_task_k[t1])
                v_e[i,0] = r2_score(h_by_task_k[t1], 
                    pca_i.inverse_transform(h_i_subspace_i),
                    multioutput='variance_weighted')
                v_e[i,1] = r2_score(h_by_task_t2s, 
                    pca_i.inverse_transform(h_j_subspace_i),
                    multioutput='variance_weighted')    
            var_ex.append(v_e)

        var_ex = np.array(var_ex)

        if self.args.do_plot and do_plot:
            pass

        return var_ex

    def compute_active_task_signal_maintenance(self, h, rules, w_td):
        """ 
        Determine how much of the *processing* is task-specific (e.g. achieved
        by specific recurrent connectivity, rather than by the direct injection
        of task signal) by subtracting task-specific input projection from
        activity at each timepoint, then doing decoding analysis.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            w_td (np.ndarray) - (N_TD x N), top-down input projection
        Returns:
            ctx_decode (np.ndarray) -- (T x 1), decoding of rule at each timept
            cross_temp_ctx_decode (np.ndarray) -- (T x T), cross-temp rule dec.

        """

        # For each task, compute the direction along which the top-down input 
        # projection forces activity; subtract this from all of those trials
        rule_inputs = w_td[-self.N_TASKS:,:]
        h_sub = h.copy()
        for r in np.unique(rules):
            inds = np.where(rules == r)[0]
            h_sub[inds] -= rule_inputs[int(r)][np.newaxis,...]

        # From the remaining activity, decode task identity
        ctx_decode = self.decode(h_sub, rules)
        cross_temp_ctx_decode = self.cross_temporal_decode(h_sub, rules)

        return ctx_decode, cross_temp_ctx_decode

    def decode_by_task(self, h, rules, y, n_pcs=args.N_PCS, do_ct=True,k=None):
        """
        Decode y separately in each task's subspace.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            sample (np.ndarray) - sample stimulus ID (B x 1)
        Returns:
            decode (np.ndarray) -- dict of (N_TASKS x T), decoding of y at 
                each timept in each epoch (PCs fit in each epoch)
            cross_temp_decode (np.ndarray) -- (N_TASKS x T x T), 
                cross-temp dec. for y in each task's subspace
        """
        # Set up arrays for recording results
        decode = []
        cross_temp_decode = []
        if type(n_pcs) is int:
            n_pcs = np.repeat(n_pcs, self.N_TASKS)

        if k is not None:
            fig, ax = plt.subplots(nrows=1, ncols=self.N_TASKS, figsize=(16,3))

        for i, t in enumerate(np.unique(rules)):

            rel_h = h[np.where(rules == t)[0]]
            rel_y = y[np.where(rules == t)[0]]

            
            rel_h_trans = PCA(int(n_pcs[i])+1).fit_transform(
                np.reshape(rel_h, (-1, self.N_NEUR)))
            rel_h_trans = np.reshape(rel_h_trans, (rel_h.shape[0], 
                rel_h.shape[1], -1))
            decode.append(self.decode(rel_h_trans, rel_y))

            if k is not None:
                r_h_t = np.reshape(rel_h_trans, (-1, int(n_pcs[i])+1))
                r_y = np.repeat(rel_y, rel_h.shape[1])
                ax[i].scatter(r_h_t[:,0], r_h_t[:,1], c=r_y)
            if do_ct:
                cross_temp_decode.append(self.cross_temporal_decode(rel_h_trans, 
                    rel_y))

        if k is not None:
            plt.tight_layout()
            fig.savefig(f"points_{k}.png", dpi=400)
            plt.close(fig)

        if do_ct:
            return decode, cross_temp_decode

        return decode

    def compute_readout_angle(self, h, rules, outputs, w_out, n_pcs=5, k=None):
        """
        Compute angle between readout axis and axis of separation in activity
        related to decision.

        Args:
            h (np.ndarray) - hidden activity tensor (B x T x N)
            rules (np.ndarray) - rule identifier for batch (B x 1)
            w_out (np.ndarray) - readout weights (N x 3)
        Returns:
            angles (np.ndarray) -- (N_TASKS, 1), angle between readout
                axis and separation axis
            dec_sep (np.ndarray) -- (N_TASKS, 1), separation b/w decision
                during test period along axis of dec separation

        """
        h_tp = h[:, self.epoch_bounds['test'],:]
        h_tp = h_tp[:,2:,:]

        # Set up records for storing angles etc.
        angles = np.zeros((self.N_TASKS,2))
        dec_sep = np.zeros(self.N_TASKS)
        read_sep = np.zeros(self.N_TASKS)

        if type(n_pcs) == int:
            n_pcs = np.repeat(n_pcs, self.N_TASKS)

        if k is not None:
            fig, ax = plt.subplots(nrows=1, ncols=self.N_TASKS, figsize=(16,3))

        for i, t in enumerate(np.unique(rules)):
            # Fit PCA for test period activity
            rel_h = np.reshape(h_tp[np.where(rules == t)[0]], (-1, self.N_NEUR))
            test_pca = PCA(n_pcs[i]).fit(rel_h)
            rel_h_red = test_pca.transform(np.reshape(h_tp[np.where(rules == t)[0]], (-1, self.N_NEUR)))
            rel_h_red = np.reshape(rel_h_red, 
                (-1, h_tp.shape[1], n_pcs[i]))

            # Identify region of readout axis illuminated by this task
            mean_test_h = np.ones_like(rel_h.mean(0))
            read_ax_1 = np.multiply(w_out[:,1], mean_test_h)
            read_ax_2 = np.multiply(w_out[:,2], mean_test_h)
            read_ax = np.vstack((read_ax_1, read_ax_2))

            # Reduce in dim. using test-period PCA
            read_ax = test_pca.transform(read_ax)

            # Fit PCA to test-period activity averaged by output required
            rel_h_o1 = h_tp[np.where((outputs == 1) & (rules == t))[0]]
            rel_h_o2 = h_tp[np.where((outputs == 2) & (rules == t))[0]]
            rel_h_o1_red = test_pca.transform(np.reshape(rel_h_o1, (-1, self.N_NEUR)))
            rel_h_o2_red = test_pca.transform(np.reshape(rel_h_o2, (-1, self.N_NEUR)))
            rel_h_o1_red_r = np.reshape(rel_h_o1_red, (rel_h_o1.shape[0], rel_h_o1.shape[1], -1))
            rel_h_o2_red_r = np.reshape(rel_h_o2_red, (rel_h_o2.shape[0], rel_h_o2.shape[1], -1))
            all_data = np.vstack(
                (rel_h_o1_red_r.mean(0), rel_h_o2_red_r.mean(0)))
            sep_pca = PCA(2).fit(all_data)
            sep_ax = sep_pca.components_[0]

            # Take top component (dec. sep. axis) and find angle w/ readout ax.
            angles[i,0] = angle_between(sep_ax.squeeze(), read_ax[0].squeeze())
            angles[i,1] = angle_between(sep_ax.squeeze(), read_ax[1].squeeze())

            # Compute decodability of decision projected onto dec. sep. axis
            dec_sep[i] = decode_output(sep_pca.transform(rel_h_o1_red), 
                sep_pca.transform(rel_h_o2_red)).mean(-1)

            # Compute decodability of decision projected onto readout axis
            read_sep[i] = decode_output(rel_h_o1_red @ read_ax.T, rel_h_o2_red @ read_ax.T).mean(-1)

            if k is not None:
                ax[i].scatter(rel_h_o1_red[:,0], rel_h_o1_red[:,1], color='red')
                ax[i].scatter(rel_h_o2_red[:,0], rel_h_o2_red[:,1], color='blue')

        if k is not None:
            plt.tight_layout()
            fig.savefig(f"points_{k}.png", dpi=500)
            plt.close(fig)

        return angles, dec_sep, read_sep


    def iterative_readout_alignment_analysis(self, h, rules, outputs, w_out, 
        n_pcs=args.N_PCS,k=None):

        if type(n_pcs) is int:
            n_pcs = np.repeat(n_pcs*2, self.N_TASKS)

        max_d_prime = [np.zeros(i) for i in n_pcs]
        readout_d_prime = [np.zeros(i) for i in n_pcs]

        # For each task: 
        for i, t in enumerate(np.unique(rules)):
            task_pca = PCA(n_pcs[i]).fit(
                np.reshape(h[np.where(rules == t)[0]], (-1, self.N_NEUR)))
            h_t = np.reshape(
                task_pca.transform(
                    np.reshape(h[np.where(rules == t)[0]], (-1, self.N_NEUR))),
                (len(np.where(rules == t)[0]), h.shape[1], n_pcs[i]))
            rel_o = outputs[np.where(rules == t)[0]]

            readout_t = task_pca.transform(w_out[:,1:].T)
            scores_1 = np.zeros((len(np.where(rel_o==1)[0])*7, n_pcs[i]))
            readouts_1 = np.zeros((len(np.where(rel_o==1)[0])*7, n_pcs[i]))

            scores_2 = np.zeros((len(np.where(rel_o==2)[0])*7, n_pcs[i]))
            readouts_2 = np.zeros((len(np.where(rel_o==2)[0])*7, n_pcs[i]))

            scores = np.zeros((h_t.shape[0]*7, n_pcs[i]))
            readouts = np.zeros((h_t.shape[0]*7, n_pcs[i]))

            # Iteratively subtract the key axes
            for j in range(n_pcs[i]):
                h_o1 = h_t[np.where(rel_o == 1)[0]]
                h_o2 = h_t[np.where(rel_o == 2)[0]]
                pca_j = PCA(1).fit(
                    np.vstack((h_o1.mean((0,1)), h_o2.mean((0,1)))))
                trans = pca_j.transform(np.reshape(h_t, (-1, n_pcs[i])))
                trans_o1 = pca_j.transform(np.reshape(h_o1[:,-7:,:], (-1, n_pcs[i])))
                trans_o2 = pca_j.transform(np.reshape(h_o2[:,-7:,:], (-1, n_pcs[i])))
                to_subtract = pca_j.inverse_transform(trans)
                to_subtract = np.reshape(to_subtract, h_t.shape)
                scores_1[:,j] = trans_o1.squeeze()
                scores_2[:,j] = trans_o2.squeeze()
                readouts_1[:,j] = pca_j.transform(np.reshape(h_o1[:,-7:,:], (-1, n_pcs[i])) * readout_t[0,:]).squeeze()
                readouts_2[:,j] = pca_j.transform(np.reshape(h_o2[:,-7:,:], (-1, n_pcs[i])) * readout_t[1,:]).squeeze()
                h_t = np.subtract(h_t, to_subtract)

                # Compute output separation by components 0 to j
                cur_comps_1 = scores_1[:,:j+1]                
                cur_comps_2 = scores_2[:,:j+1]
                max_d_prime[i][j] = decode_output(cur_comps_1, cur_comps_2).mean(-1)

                # Compute output separation when projected onto readout 
                h_o1_readout = readouts_1[:,:j+1]                
                h_o2_readout = readouts_2[:,:j+1]
                readout_d_prime[i][j] = decode_output(h_o1_readout, h_o2_readout).mean(-1)

        return readout_d_prime, max_d_prime


class ResultsAnalyzer:

    def __init__(self, args, fns, net_fns, iters):

        self.args = args
        self.fns = fns 
        self.net_fns = net_fns
        self.iters = iters
        self.training_data, self.trained_data = self.load_data(fns, net_fns, 
            iters)

        self.epoch_bounds = {'dead': [0, 200],
                       'fix': [200, 500],
                       'sample': [500,800],
                       'delay': [800,1800],
                       'test': [1800,2100]}
        self.task_names = ['DMS', 'DMS+distractor', 'DMRS180', 'DMC', 'Delay-Go', 'ProWM', 'RetroWM']
        self.N_TASKS = 7
        self.N_SAMPLE = 8
        self.TRIAL_LEN = 2100

        return

    def get_colors(self, n):
        colors = plt.cm.jet(np.linspace(0,1,n))
        lightdark_colors = plt.cm.tab20(np.linspace(0, 1, n*2))
        light_colors = lightdark_colors[::2]
        dark_colors = lightdark_colors[1::2]

        return colors, light_colors, dark_colors

    def load_data(self, fns, net_fns, iters):
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
        training_related = ['accs', 'exp_var_task_sub', 'act_seps',
            'output_task_dep', 'shared_ax_task_decode', 'shared_ax_separation',
            'var_exp_shared_response_subs', 'out_dec_shared_response_subs', 
            'time_dependent_sequestration', '']
        training_data = defaultdict(list)
        trained_data  = defaultdict(list)
        
        for fn, net_fn in zip(fns, net_fns):
            print(fn)
            d = pickle.load(open(fn, 'rb'))
            d1 = pickle.load(open(net_fn, 'rb'))
            for k, v in d.items():
                v = np.array(v)
                if len(v) == 0:
                    continue
                # Load in the training related data
                if v.shape[0] == len(iters):
                    training_data[k].append(v)
                    trained_data[k].append(v[-1:])
                else:
                    trained_data[k].append(v)
            trained_data['sample_decoding'] = d1['sample_decoding']

        # Bind all into numpy arrays
        training_data = {k: np.array(v).squeeze() for k, v in 
            training_data.items()}
        trained_data  = {k: np.array(v).squeeze() for k, v in
            trained_data.items()}

        return training_data, trained_data

    def generate_all_figures(self, to_plot=[4]):
        
        ########################################################################
        # I. Networks inhabit a stable "task subspace", in which activity for 
        # different tasks remains continuously and stably discriminable
        ########################################################################
        if 0 in to_plot:
            self.plot_figure_i()

        ########################################################################
        # II. Computations that are shared across tasks can be carried out in 
        # shared subspaces, even as network activity remains task-specific in
        # the task subspace
        ########################################################################
        if 1 in to_plot:
            self.plot_figure_ii()

        ########################################################################
        # III. Networks successfully solve multiple tasks by generating 
        # decision-aligned variation within the task subspace (NEED TO CHECK THIS!!!),
        # and by aligning the output readouts with this variation
        ########################################################################
        if 2 in to_plot:
            self.plot_figure_iii()

        ########################################################################
        # IV. Networks solve tasks by importing activity into a shared output-
        # potent subspace
        ########################################################################
        if 3 in to_plot:
            self.plot_figure_iv()
        
        ########################################################################
        # V. Networks solve these tasks by parcellating state space into task-
        # specific regions
        ########################################################################
        if 4 in to_plot:
            self.plot_figure_v()

        ########################################################################
        # VI. Networks learn to solve tasks by progressively shunting activity
        # toward subspaces that can sustain task-appropriate dynamics, and 
        # by aligning the readout vectors with this variation.
        ########################################################################
        if 5 in to_plot:
            self.plot_figure_vi()

        # AA. Decoding performance from sample decode @ the beginning of training

        return figures

    def plot_figure_i(self):

        fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(16,3))
        colors, light_colors, dark_colors = self.get_colors(2)

        # A. Decoding task from activity (raw)
        raw_ctx_mean = np.mean(self.trained_data['context_decode_raw'], 0)
        raw_ctx_sd = np.std(self.trained_data['context_decode_raw'], 0)
        x = np.arange(0, self.TRIAL_LEN, self.TRIAL_LEN // len(raw_ctx_mean))
        ax[0].plot(x, raw_ctx_mean, linewidth=2, label='Task decode')
        ax[0].plot(x, [1 / self.N_TASKS]*len(raw_ctx_mean), 
            linestyle='--', label='Chance')
        ax[0].fill_between(x, (raw_ctx_mean - raw_ctx_sd), 
            (raw_ctx_mean + raw_ctx_sd), alpha=0.1)
        ax[0].set(xlabel="Time", ylabel="Task decode (raw)")
        ax[0].legend()

        # B. Decoding task from activity (raw, cross-temp.; ex. net)
        # Identify network w/ maximally stable cross-temporal decode
        raw_cross_temp_stab = self.compute_cross_temporal_stability(
            self.trained_data['cross_temp_ctx_raw'])
        max_stab_raw = np.argmax(raw_cross_temp_stab)
        ct_raw_max = self.trained_data['cross_temp_ctx_raw'][max_stab_raw]
        im = ax[1].imshow(ct_raw_max,
                vmin=1.0 / self.N_TASKS,
                vmax=1.0)
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
        ax[1].set(xlabel="Tr. timestep", ylabel="Te. timestep")
        
        # C. Decoding task from activity (task-dep. subs.)
        td_ctx_mean = np.mean(self.trained_data['context_decode_proj_td'], 0)
        td_ctx_sd = np.std(self.trained_data['context_decode_proj_td'], 0)
        x = np.arange(0, self.TRIAL_LEN, self.TRIAL_LEN // len(td_ctx_mean))
        ax[2].plot(x, td_ctx_mean, linewidth=2, label='Task decode')
        ax[2].plot(x, [1 / self.N_TASKS]*len(td_ctx_mean), 
            linestyle='--', label='Chance')
        ax[2].fill_between(x, (td_ctx_mean - td_ctx_sd), 
            (td_ctx_mean + td_ctx_sd), alpha=0.1)
        ax[2].set(xlabel="Time", ylabel="Task decode (task subs.)")
        ax[2].legend()

        # D. Decoding task from activity (task-dep. subs., cross-temp.; ex. net)
        td_cross_temp_stab = self.compute_cross_temporal_stability(
            self.trained_data['cross_temp_ctx_td'])
        max_stab_td = np.argmax(td_cross_temp_stab)
        ct_td_max = self.trained_data['cross_temp_ctx_td'][max_stab_td]
        im = ax[3].imshow(ct_td_max,
                vmin=1.0 / self.N_TASKS,
                vmax=1.0)
        divider = make_axes_locatable(ax[3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
        ax[3].set(xlabel="Tr. timestep", ylabel="Te. timestep")

        # E. Quantifying cross-temp. decoder stability (raw vs. task-dep. subs.)
        raw_stability_mean = np.mean(raw_cross_temp_stab)
        raw_stability_sd = np.std(raw_cross_temp_stab)
        td_stability_mean = np.mean(td_cross_temp_stab)
        td_stability_sd = np.std(td_cross_temp_stab)
        x = [0, 1]
        y = [raw_stability_mean, td_stability_mean]
        yerr = [raw_stability_sd, td_stability_sd]
        ax[4].bar(x, y, color=colors, yerr=yerr)
        ax[4].set(xticklabels=['', 'Raw', 'Task subs.'], 
                  xlabel='Subspace',
                  ylabel='Task decode stability')

        # F. Examining dimensionality of the task subspace (var. ex. as fxn of
        # number of components)
        exp_var = self.trained_data['context_pcs_exp_var']
        exp_var_mean = exp_var.mean(0)
        exp_var_std = exp_var.std(0)
        ax[5].plot(np.arange(len(exp_var_mean)), exp_var_mean, linewidth=2)
        ax[5].fill_between(np.arange(len(exp_var_mean)), 
            (exp_var_mean + exp_var_std), (exp_var_mean - exp_var_std),
            alpha=0.1)
        ax[5].set(xlabel="Dimension #", ylabel="Frac. var. exp.")

        plt.tight_layout()
        fig.savefig("fig_pt_i.png", dpi=400)
        plt.close(fig)

        return

    def plot_figure_ii(self):

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,6))
        colors, light_colors, dark_colors = self.get_colors(3)

        # A. Decoding sample from activity (shared subs.)
        ti_sam_mean = np.mean(self.trained_data['sample_decode_proj_ti'], 0)
        ti_sam_sd = np.std(self.trained_data['sample_decode_proj_ti'], 0)
        x = np.arange(0, self.TRIAL_LEN, self.TRIAL_LEN // len(ti_sam_mean))
        ax[0,0].plot(x, ti_sam_mean, linewidth=2, label='Sample decode')
        ax[0,0].plot(x, [1 / self.N_SAMPLE]*len(ti_sam_mean), 
            linestyle='--', label='Chance')
        ax[0,0].fill_between(x, (ti_sam_mean - ti_sam_sd), 
            (ti_sam_mean + ti_sam_sd), alpha=0.1)
        ax[0,0].set(xlabel="Time", ylabel="Sample decode (shared subs.)")
        ax[0,0].legend()

        # H. Decoding sample from activity (shared subs., cross-temp)
        ti_cross_temp_stab = self.compute_cross_temporal_stability(
            self.trained_data['cross_temp_sam_ti'])
        max_stab_ti = np.argmax(td_cross_temp_stab)
        ct_ti_max = self.trained_data['cross_temp_sam_ti'][max_stab_ti]
        im = ax[1,0].imshow(ct_ti_max,
                vmin=1.0 / self.N_SAMPLE,
                vmax=1.0)
        divider = make_axes_locatable(ax[1,0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
        ax[1,0].set(xlabel="Tr. timestep", ylabel="Te. timestep")

        # I. Decoding sample from activity (task subs.)
        td_sam_mean = np.mean(self.trained_data['sample_decode_proj_td'], 0)
        td_sam_sd = np.std(self.trained_data['sample_decode_proj_td'], 0)
        x = np.arange(0, self.TRIAL_LEN, self.TRIAL_LEN // len(td_sam_mean))
        ax[0,1].plot(x, td_sam_mean, linewidth=2, label='Sample decode')
        ax[0,1].plot(x, [1 / self.N_SAMPLE]*len(td_sam_mean), linestyle='--', label='Chance')
        ax[0,1].fill_between(x, (td_sam_mean - td_sam_sd), 
            (td_sam_mean + td_sam_sd), alpha=0.1)
        ax[0,1].set(xlabel="Time", ylabel="Sample decode (task subs.)")
        ax[0,1].legend()

        # J. Decoding sample from activity (task-dep. subs., cross-temp)
        td_cross_temp_stab = self.compute_cross_temporal_stability(
            self.trained_data['cross_temp_sam_td'])
        max_stab_td = np.argmax(td_cross_temp_stab)  
        ct_td_max = self.trained_data['cross_temp_sam_td'][max_stab_td]
        im = ax[1,1].imshow(ct_td_max,
                vmin=1.0 / len(np.unique(self.N_SAMPLE)),
                vmax=1.0)
        divider = make_axes_locatable(ax[1,1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
        ax[1,1].set(xlabel="Tr. timestep", ylabel="Te. timestep")

        # K. Quantifying cross-temp. decoder stability (raw, task-dep, shared)
        raw_stability_mean = np.mean(raw_cross_temp_stab)
        raw_stability_sd = np.std(raw_cross_temp_stab)
        ti_stability_mean = np.mean(ti_cross_temp_stab)
        ti_stability_sd = np.std(ti_cross_temp_stab)
        td_stability_mean = np.mean(td_cross_temp_stab)
        td_stability_sd = np.std(td_cross_temp_stab)
        x = [0, 1, 2]
        y = [raw_stability_mean, ti_stability_mean, td_stability_mean]
        yerr = [raw_stability_sd, ti_stability_sd, td_stability_sd]
        ax[0,2].bar(x, y, color=colors, yerr=yerr)
        ax[0,2].set(xticks=x, 
                  xticklabels=['Raw', 'Shared subs.', 'Task subs.'], 
                  xlabel='Subspace',
                  ylabel='Task decode stability')

        # L. Decoding sample from activity (raw, cross-temp)
        raw_cross_temp_stab = self.compute_cross_temporal_stability(
            self.trained_data['cross_temp_sam_raw'])
        max_stab_raw = np.argmax(raw_cross_temp_stab)
        ct_raw_max = self.trained_data['cross_temp_sam_raw'][max_stab_raw]
        im = ax[1,2].imshow(ct_raw_max,
                vmin=1.0 / self.N_SAMPLE,
                vmax=1.0)
        divider = make_axes_locatable(ax[1,2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
        ax[1,2].set(xlabel="Tr. timestep", ylabel="Te. timestep")

        plt.tight_layout()
        fig.savefig("fig_pt_ii.png", dpi=400)
        plt.close(fig)

        return


    def plot_figure_iii(self):

        fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(25,3))


        # O. Plot sequestration as a function of accuracy (trained networks)
        seq = self.compute_sequestration_scores(
            self.training_data['exp_var_task_sub'])
        seq_final = seq[:, -1, ...].flatten()
        acc_final = self.training_data['accs'][:, -1, ...].flatten()
        corr, p = self.compute_correlation(acc_final, seq_final)
        ax[0].scatter(acc_final, seq_final, label='$R$ = ' + f"{corr:.2f} (P = {p:.2e})")
        ax[0].legend()
        ax[0].set(xlabel='Accuracy', ylabel='Sequestration')

        # P. Plot sequestration progression through training (norm to 
        # initial value in first iteration)
        ax[1].plot(iters, seq.mean((0,2)), linewidth=2, color='black')
        ax[1].plot(np.tile(iters, (self.training_data['accs'].shape[0], 1)).T,
            seq.mean(-1).T, alpha=0.1, color='black')
        ax[1].set(xlabel='Iteration', ylabel='Sequestration')

        # Q. Plot distribution of correlations between sequestration and 
        # accuracy through training of each task
        seq_tasknet_pairs = np.transpose(seq, (0, 2, 1))
        seq_tasknet_pairs = np.reshape(seq_tasknet_pairs, 
            (-1, seq_tasknet_pairs.shape[-1]))
        acc_tasknet_pairs = np.transpose(self.training_data['accs'],
            (0, 2, 1))
        acc_tasknet_pairs = np.reshape(acc_tasknet_pairs,
            (-1, acc_tasknet_pairs.shape[-1]))
        corr_distribution = [self.compute_correlation(s, a)[0] 
            for s, a in zip(seq_tasknet_pairs, acc_tasknet_pairs)]
        ax[2].hist(corr_distribution, 20, density=True)
        ax[2].set(xlabel="Spearman's R (sequestration vs. accuracy)",
            ylabel='Prob. density')

        # R. Plot separation of activity progression through training
        act_sep = self.training_data['alignments'][...,0].squeeze()
        out_sep = self.training_data['alignments'][...,1].squeeze()
        ax[3].plot(iters, act_sep.mean((0,2)), linewidth=2, color='black')
        ax[3].plot(np.tile(iters, (self.training_data['accs'].shape[0], 1)).T,
            act_sep.mean(-1).T, alpha=0.1, color='black')
        ax[3].set(xlabel='Iteration', ylabel=r'Separation ($D^\prime$)')

        # S. Plot separation of activity projected onto output axis (alignment)
        ax[4].plot(iters, out_sep.mean((0,2)), linewidth=2, color='black')
        ax[4].plot(np.tile(iters, (self.training_data['accs'].shape[0], 1)).T,
            out_sep.mean(-1).T, alpha=0.1, color='black')
        ax[4].set(xlabel='Iteration', ylabel=r'Separation ($D^\prime$)')

        # T. Plot distribution of correlations b/w activity separation
        # and accuracy through training of each task
        sep_tasknet_pairs = np.transpose(act_sep, (0, 2, 1))
        sep_tasknet_pairs = np.reshape(sep_tasknet_pairs, 
            (-1, sep_tasknet_pairs.shape[-1]))
        corr_distribution = [self.compute_correlation(s, a)[0] 
            for s, a in zip(sep_tasknet_pairs, acc_tasknet_pairs)]
        ax[5].hist(corr_distribution, 20, density=True)
        ax[5].set(xlabel="Spearman's R\n(activity sep. vs. acc.)",
            ylabel='Prob. density')

        # U. Plot distribution of correlations b/w activity separation
        # when projected onto output and accuracy through training of each task
        out_tasknet_pairs = np.transpose(out_sep, (0, 2, 1))
        out_tasknet_pairs = np.reshape(out_tasknet_pairs, 
            (-1, out_tasknet_pairs.shape[-1]))
        corr_distribution = [self.compute_correlation(o, a)[0] 
            for o, a in zip(out_tasknet_pairs, acc_tasknet_pairs)]
        ax[6].hist(corr_distribution, 20, density=True)
        ax[6].set(xlabel="Spearman's R\n(output sep. vs. acc.)",
            ylabel='Prob. density')

        # V. Plot/compute correlations at each timepoint of training, across
        # all network/task pairs, between sequestration and accuracy on the task
        # (Nick's suggested analysis)
        corr_timecourse = [self.compute_correlation(s_t, a_t)[0]
            for s_t, a_t in zip(seq_tasknet_pairs.T, acc_tasknet_pairs.T)]
        ax[7].scatter(self.iters, corr_timecourse)

        corr_timecourse = [self.compute_correlation(s_t, a_t)[0]
            for s_t, a_t in zip(sep_tasknet_pairs.T, acc_tasknet_pairs.T)]
        ax[8].scatter(self.iters, corr_timecourse)

        corr_timecourse = [self.compute_correlation(s_t, a_t)[0]
            for s_t, a_t in zip(out_tasknet_pairs.T, acc_tasknet_pairs.T)]
        ax[8].scatter(self.iters, corr_timecourse)

        plt.tight_layout()
        fig.savefig("fig_pt_iii.png", dpi=400)
        plt.close(fig)

        return

    def plot_figure_iv(self):

        fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(18,3))

        # W. Separation of activity when projected onto primary axes of
        # decision-related variation *for each task* (if high for off-diagonal
        # entries, evidence for shared output subspace)
        o_t_d = self.trained_data['output_task_dep']
        c = ax[0].imshow(o_t_d[1].squeeze(), vmin=0, vmax=1)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(c, cax=cax)

        # X. Decode task during test period when projected onto shared axes of 
        # activity separation (if low, evidence for shared output subspace)
        s_a_t_d = self.trained_data['shared_ax_task_decode']
        y = s_a_t_d.mean((0,-1))
        yerr = s_a_t_d.std((0, -1))
        x = range(len(y))
        print(s_a_t_d.shape)
        ax[1].plot(s_a_t_d.mean((0,-1)), color='black')
        ax[1].plot(s_a_t_d.mean(-1).T, color='black', alpha=0.1)
        ax[1].plot(x, np.repeat(1/self.N_TASKS, len(x)), linestyle='--', 
            color='black', label='Chance')
        ax[1].fill_between(x, (y - yerr), 
            (y + yerr), alpha=0.1, color='black')
        ax[1].legend()

        # Y. Separation of activity when projected onto shared axes of 
        # activity separation (if high, evidence for shared output subspace)
        s_a_s = self.trained_data['shared_ax_separation']
        x = range(s_a_s.shape[-1])
        for i, a in enumerate(s_a_s):
            acc_i = self.trained_data['accs'][i].mean()

            color = 'black'
            if acc_i > 0.8:
                color = 'red'
            if acc_i > 0.85: 
                color = 'green'
            if acc_i > 0.9:
                color = 'purple'
            ax[2].plot(x, a[0], label=f"{acc_i:.2f}",alpha=0.4,color=color)
            ax[3].plot(x, a[1], label=f"{acc_i:.2f}",alpha=0.4,color=color)
        #ax[2].legend()
        #ax[3].legend()

        # Y. Identify response subspace directly (variance explained of all
        # activity when projected onto these components);
        v_e_s_r_s = self.trained_data['var_exp_shared_response_subs']
        print(v_e_s_r_s.shape)
        ax[4].hist(v_e_s_r_s[np.where(v_e_s_r_s > -10)])
        
        # Z. Decoding when projected into shared response subspace
        o_d_s_r_s = self.trained_data['out_dec_shared_response_subs']
        x = range(o_d_s_r_s.shape[1])
        for i, a in enumerate(o_d_s_r_s):
            acc_i = self.trained_data['accs'][i].mean()
            color = 'black'
            if acc_i > 0.8:
                color = 'red'
            if acc_i > 0.85: 
                color = 'green'
            if acc_i > 0.9:
                color = 'purple'
            ax[5].plot(x, a, label=f"{self.trained_data['accs'][i].mean():.2f}",color=color,alpha=0.4)


        # Z. Time-dependent sequestration analysis (is activity less sequestered
        # during the test period than it is during the rest of the task?)
        # --> May be issues with this; do we need to average across sample 
        # stimuli for each task before identifying PCs?

        plt.tight_layout()
        fig.savefig("fig_pt_iv.png", dpi=400)
        plt.close(fig)

        return

    def plot_figure_v(self):

        fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(20,3))
        colors, light_colors, dark_colors = self.get_colors(self.N_TASKS)

        # A. Single network: PCA on all activity, color by task
        # (qualitatively, task ID is dominant signal in activity)
        # (DONT HAVE THIS YET; MAYBE NOT NEEDED)


        # B. Same network: decode of task ID (raw), cross-temporal 
        # (task subspace) -- task is a dominant signal, and can be stably
        # discriminated in same subspace for duration of trial
        raw_ctx_mean = np.mean(self.trained_data['context_decode_raw'], 0)
        raw_ctx_sd = np.std(self.trained_data['context_decode_raw'], 0)
        x = np.arange(0, self.TRIAL_LEN, self.TRIAL_LEN // len(raw_ctx_mean))
        ax[1].plot(x, raw_ctx_mean, linewidth=2, label='Task decode')
        ax[1].plot(x, [1 / self.N_TASKS]*len(raw_ctx_mean), 
            linestyle='--', label='Chance')
        ax[1].fill_between(x, (raw_ctx_mean - raw_ctx_sd), 
            (raw_ctx_mean + raw_ctx_sd), alpha=0.1)
        ax[1].set(xlabel="Time", ylabel="Task decode (raw)")
        ax[1].legend()

        td_cross_temp_stab = self.compute_cross_temporal_stability(
            self.trained_data['cross_temp_ctx_td'])
        max_stab_td = np.argmax(td_cross_temp_stab)
        ct_td_max = self.trained_data['cross_temp_ctx_td'][max_stab_td]
        im = ax[2].imshow(ct_td_max,
                vmin=1.0 / self.N_TASKS,
                vmax=1.0)
        divider = make_axes_locatable(ax[3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax)
        ax[2].set(xlabel="Tr. timestep", ylabel="Te. timestep")


        # C. Mean time-dependent sequestration (network activity is 
        # highly sequestered for the duration of each trial)
        td_seq = self.trained_data['time_dependent_sequestration']
        seq_by_epoch = self.compute_sequestration_scores(td_seq)
        x = [0, 1, 2, 3, 4]
        y = np.mean(seq_by_epoch, axis=0)
        yerr = np.std(seq_by_epoch, axis=0)
        ax[3].bar(x, y, color=colors, yerr=yerr)
        ax[3].set(xticks=x, 
                  xticklabels=['Dead', 'Fix', 'Sample', 'Delay', 'Test'], 
                  xlabel='Epoch',
                  ylabel='Sequestration')

        # D. Output task-dependence: variation related to decision in 
        # each task's subspace is not aligned with variation related
        # to decision for other tasks
        o_t_d = self.trained_data['output_task_dep']
        x = [0, 1, 2, 3, 4, 5, 6]
        deps = np.divide(o_t_d[:,:,0].squeeze(), o_t_d[:,:,1].squeeze())
        y = np.mean(deps, 0)
        yerr = np.std(deps, 0)
        ax[4].bar(x, y, color=colors, yerr=yerr)
        ax[4].set(xticks=x, 
                  xticklabels=self.task_names, 
                  xlabel='Task',
                  ylabel='Output task-dependence')

        # E. Task decodability from projection onto output axes
        out_task_dec = self.trained_data['out_task_dec']
        print(out_task_dec.shape)
        y = np.mean(out_task_dec, axis=0)
        yerr = np.std(out_task_dec, axis=0)
        x = np.arange(self.epoch_bounds['test'][0]+40, self.epoch_bounds['test'][1], 260 // len(y))
        ax[5].plot(x, y, linewidth=2, label='Task decode')
        ax[5].plot(x, [1 / self.N_TASKS]*len(y), 
            linestyle='--', label='Chance')
        ax[5].fill_between(x, (y - yerr), 
            (y + yerr), alpha=0.1)
        ax[5].set(xlabel="Time", ylabel="Task decode (raw)")
        ax[5].legend()

        # F. Sample decode in each task subspace is related to task
        # performance (strong evidence for solution by sequestration:
        # if the necessary information isn't found *specifically* in the 
        # principal axes of variation for that task, then the task
        # cannot be solved.)


        plt.tight_layout()
        fig.savefig("fig_pt_v.png", dpi=400)
        plt.close(fig)

        return

    def plot_figure_vi(self):

        fig, ax = plt.subplots(nrows=1, ncols=7, figsize=(18,3))

        # A. 

        return



    def predict_task_performance(self, X, y, n_repetitions=100, k_folds=4):
        """
        Given features that measure network activity during each task,
        predict task performance.

        Args:
            X (np.ndarray) -- (N_NETS*N_TASKS x N_FEAT), measures of activity
            y (np.ndarray) -- (N_NETS*N_TASKS x 1), performance on each task
        Returns:
            scores (np.ndarray) -- (N_REP x K_FOLD) reg. scores
            coef (np.ndarray) -- (N_REP x K_FOLD x N_FEAT) coefficients for 
                predicting accuracy from features
        """
        scores = np.zeros((n_repetitions, k_folds))
        skf = StratifiedKFold(n_splits=k_folds)
        lso = Lasso()
        coef = np.zeros((n_repetitions, k_folds, X.shape[1]))
        for rep in range(n_repetitions):
            for i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                y_tr, y_te = y[tr_idx], y[te_idx]
                X_tr, X_te = X[tr_idx], X[te_idx]
                lso.fit(X_tr, y_tr)
                scores[rep, i] = lso.score(X_te, y_te)
                coef[rep, i, :] = lso.sparse_coef_.squeeze()

        return scores, coef

    def adjust_sequestration_scores(self, seq):
        """
        Change range of sequestration/activity separation to span 0 to 1
        with values nearer to 1 = maximal separation
        """
        for i in range(seq.shape[1]):
            if seq[:,i].min() < 0:
                seq[:,i] += abs(seq[:,].min())
            seq[:,i] /= np.amax(seq[:,i]) # Add |most negative value|, then divide by largest
            seq[:,i] = 1 - seq[:,i]
        return seq

    def compute_correlation(self, a, b):
        """ 
        Compute correlation (Spearman's R) between variables a and b.

        Args:
            a (np.ndarray, (n x 1))
            b (np.ndarray, (n x 1))
        Returns:
            corr (float) -- Spearman's R
            p (float) -- p-value of correlation
        """
        corr, p = stats.spearmanr(a, b)
        return corr, p

    def compute_sequestration_scores_pairwise(self, var_ex):
        """
        Compute sequestration score for each task on the basis of how much
        variance in its activity can be explained by the principal axes of
        other tasks.

        Args:
            exp_vars (np.ndarray) -- (N_NETS x N_TASKS x N_TASKS), exp. var.
                of each task's activity from principal axes of other tasks
        Returns:
            seq (np.ndarray) -- (N_NETS x N_TASKS)
        """
        off_diag = np.float32(~np.eye(self.N_TASKS, dtype=bool))
        seq = []
        for v in var_ex:
            s = np.array([np.mean(i*off_diag,axis=1) for i in v])
            seq.append(self.adjust_sequestration_scores(s))

        return np.array(seq)

    def compute_sequestration_scores(self, seq_sc):

        seq = np.zeros((seq_sc.shape[0], seq_sc.shape[1]))

        for net in range(seq_sc.shape[0]):
            for epoch in range(seq_sc.shape[1]):
                
                # Take mean seq across tasks
                seq[net, epoch] = np.mean(
                    np.maximum(seq_sc[net,epoch,:,0], 0.) - \
                    np.maximum(seq_sc[net,epoch,:,1],0.))

        return seq

    def compute_cross_temporal_stability(self, ct):
        """
        Compute stability of cross-temporal decoding analysis -- mean of the
        off-diagonal entries * (1 - variance of off-diagonal).

        Args:
            ct (np.ndarray) -- (N_NETS x N_TIME x N_TIME), cross-temp. decode
        Returns:
            stability (np.ndarray) -- stability score for each network
        """
        off_diag = np.float32(~np.eye(ct.shape[1], dtype=bool))
        ct_means = np.array([np.mean(i*off_diag) for i in ct])
        ct_vars = np.array([1 - np.var(i*off_diag) for i in ct])

        return np.multiply(ct_means, ct_vars)


def d_prime(A, B):
    """
    D-prime for unequal variances.

    Args: 
        A, B (np.ndarray) -- both (k x 1)
    Returns:
        d_prime (float) -- d-prime      
    """
    num = np.sqrt(2) * abs(A.mean() - B.mean())
    denom = np.sqrt(A.var() + B.var())

    return num / denom

def multid_d_prime(xx,yy):
    #c_A = np.cov(A)
    #c_B = np.cov(B)
    #s_rms = (c_A + c_B / 2) ** 0.5
    #m_A = np.mean(A, axis=0)
    #m_B = np.mean(B, axis=0)
    #print(A.shape, B.shape, s_rms.shape, m_A.shape, m_B.shape)
    #D = np.linalg.norm(np.linalg.pinv(s_rms) * (m_A - m_B))
    X = np.vstack([xx.T,yy.T])
    V = np.cov(X.T)
    VI = np.linalg.inv(V)
    D = np.diag(np.sqrt(np.dot(np.dot((xx.T-yy.T),VI),(xx.T-yy.T).T)))
    return D

def decode_output(X1, X2, k_folds=4):
    """
    Decode output, where features segregated by label.

    Args:
        X1 (np.ndarray) - activity for output-1 trials.
        X2 (np.ndarray) - activity for output-2 trials.
    Returns:
        acc (np.ndarray) - (K_FOLDS x 1)
    """
    skf = StratifiedKFold(n_splits=k_folds)
    scores = np.zeros((k_folds))
    svm = SVC(C=1.0, kernel='linear', max_iter=1000, 
            decision_function_shape='ovr', shrinking=False, tol=1e-3)
    if len(X1.shape) < 2:
        X1 = X1[:,np.newaxis]
    if len(X2.shape) < 2:
        X2 = X2[:,np.newaxis]
    X = np.vstack((X1, X2))
    y = np.vstack((np.zeros((X1.shape[0],1)), np.ones((X2.shape[0],1))))
    for i, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        y_tr, y_te = y[tr_idx], y[te_idx]
        X_tr, X_te = X[tr_idx, :], X[te_idx, :]
        ss = StandardScaler()
        ss.fit(X_tr)
        X_tr = ss.transform(X_tr)
        X_te = ss.transform(X_te)
        svm.fit(X_tr, y_tr)
        scores[i] = svm.score(X_te, y_te)

    return scores

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
    return glob.glob(data_path + "*acc=0.9*_tr.pkl")


if __name__ == "__main__":
    fns = get_fns()
    if not os.path.exists(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.exists(args.analysis_save_path):
        os.makedirs(args.analysis_save_path)

    task_titles = ["DMS", "DMS_distractor", "DMRS180", 
        "DMC", "DelayGo", "ProWM", "RetroWM"]
    iters = np.concatenate((np.arange(20), 
        np.arange(20, 100, 10), 
        np.arange(100, 200, 25),[200])).flatten()

    results_fns = []
    net_fns = []

    for fn in fns:
        if not "0.9001" in fn:
            continue
        print(fn)

        net_analyzer = NetworkAnalyzer(fn, args)
        #if not hasattr(net_analyzer, 'h'):
        #    continue

        # Perform specified analyses
        to_do = [32]#[2,5,6,13,14,25,29,31]
        results, results_fn = net_analyzer.perform_all_analyses(to_do)
        results_fns.append(results_fn)
        net_fns.append(fn)

    # Perform summary analyses
    ra = ResultsAnalyzer(args, results_fns, net_fns, iters)
    ra.generate_all_figures()
    

    