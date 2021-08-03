import numpy as np
from tasks import Task

class ProWM(Task.Task):
    # TODO: must resolve how outputs are set (e.g. in case of delay go and dms being trained in same net)
    # currently, assuming fix/match/nonmatch are 0/1/2

    def __init__(self, task_name, rule_id, var_delay, dt, tuning, timing, shape, misc):

        # Initialize from superclass Task
        super().__init__(task_name, rule_id, var_delay, dt, tuning, timing, shape, misc)
        self.categorization = misc['categorization']

    def _get_trial_info(self, batch_size):
        return super()._get_trial_info(batch_size)

    def generate_trials(self, batch_size, test_mode=False):
        """
        Generate ProWM trials (n_RFs stimuli)
        Modeled off specifically prospective trials from the task 
        in the Buschman paper, but with the allowance for k different 
        sample stimuli to be presented at once.
        """

        # Set up trial_info with ProWM-specific entries, and amending the entry for sample
        trial_info = self._get_trial_info(batch_size)
        trial_info['cue'] = np.zeros((batch_size), dtype=np.float32)

        for i in range(batch_size):

            # Determine sample stimulus, RFs, pro/retro
            sample_dirs = np.random.choice(self.n_motion_dirs, self.n_sample, replace=True)
            test_cue    = np.random.choice(self.n_RFs)
            cued_dir    = sample_dirs[test_cue]
            catch       = np.random.rand() < self.catch_trial_pct

            # Establish task timings
            fix_bounds      = [0, (self.dead_time + self.fix_time)]
            rule_bounds     = [0, self.trial_length]
            sample_bounds   = [fix_bounds[-1], fix_bounds[-1] + self.sample_time]
            delay_bounds    = [sample_bounds[-1], sample_bounds[-1] + self.delay_time]
            test_bounds     = [delay_bounds[-1], delay_bounds[-1] + self.test_time]
            response_bounds = test_bounds
            cue_bounds = [fix_bounds[-1] - self.cue_time, self.trial_length]

            # Set mask at critical periods
            trial_info['train_mask'][i, test_bounds[0]:test_bounds[0]+self.mask_duration] = 0
            trial_info['train_mask'][:, range(0, self.dead_time)] = 0
            trial_info['train_mask'][i, range(*test_bounds)] *= self.test_cost_multiplier
            trial_info['train_mask'][i, test_bounds[-1]:] = 0

            # Generate inputs
            sample_inputs = []
            for k, s_k in enumerate(sample_dirs):
                sample_inputs.append(np.reshape(self.motion_tuning[:, k, s_k],(1,-1)))
            sample_input = np.sum(np.stack(sample_inputs), axis=0)

            fix_input  = int(self.n_fix_tuned > 0) * np.reshape(self.fix_tuning[:,0],(1,-1))
            rule_input = int(self.n_rule_tuned > 0) * np.reshape(self.rule_tuning[:,self.rule_id],(1,-1))
            cue_input  = int(self.n_cue_tuned > 0) * np.reshape(self.cue_tuning[:,test_cue],(1,-1))
            trial_info['neural_input'][i, range(*sample_bounds), :]    += sample_input
            trial_info['neural_input'][i, range(0, test_bounds[0]), :] += fix_input
            trial_info['neural_input'][i, range(*rule_bounds), :]      += rule_input
            trial_info['neural_input'][i, range(*cue_bounds), :]       += cue_input

            if self.categorization:
                N = self.n_motion_dirs // 2
                resp_idx = 1 + cued_dir // N
            else:
                resp_idx = 1 + cued_dir

            # Generate outputs
            trial_info['desired_output'][i, range(0, test_bounds[0]), 0] = 1.
            if not catch:
                trial_info['desired_output'][i, range(*test_bounds), resp_idx] = 1.
            else:
                trial_info['desired_output'][i, range(*test_bounds), 0] = 1.

            # Generate reward matrix, shape (T, output_size)
            reward_matrix = np.zeros((self.trial_length, self.n_output), dtype=np.float32)
            reward_matrix[range(response_bounds[0]), 1:] = self.fix_break_penalty
            if not catch:
                reward_matrix[range(*response_bounds), 1:] = self.wrong_choice_penalty
                reward_matrix[range(*response_bounds), resp_idx] = self.correct_choice_reward
                reward_matrix[-1, 0] = self.fix_break_penalty
            else:
                reward_matrix[-1, 0] = self.correct_choice_reward
            trial_info['reward_matrix'][i,...] = reward_matrix

            # Record trial information
            trial_info['sample'][i,:] = sample_dirs
            trial_info['catch'][i]  = bool(catch)
            trial_info['rule'][i]   = self.rule_id
            trial_info['cue'][i]    = test_cue
            timing_dict = {'fix_bounds'     : fix_bounds,
                           'sample_bounds'  : sample_bounds,
                           'delay_bounds'   : delay_bounds,
                           'test_bounds'    : test_bounds,
                           'cue_bounds'     : cue_bounds,
                           'response_bounds': response_bounds}
            trial_info['timing'].append(timing_dict)


        return trial_info
