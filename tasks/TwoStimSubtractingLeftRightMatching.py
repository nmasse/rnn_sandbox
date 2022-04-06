import numpy as np
from tasks import Task

class TwoStimSubtractingLeftRightMatching(Task.Task):
    # TODO: must resolve how outputs are set (e.g. in case of delay go and dms being trained in same net)
    # currently, assuming fix/match/nonmatch are 0/1/2

    def __init__(self, task_name, rule_id, var_delay, dt, tuning, timing, shape, misc):

        # Initialize from superclass Task
        super().__init__(task_name, rule_id, var_delay, dt, tuning, timing, shape, misc)

        # Hard-set some parameters (shouldn't be changeable across all new tasks)
        self.n_sample = 2

    def _get_trial_info(self, batch_size):
        return super()._get_trial_info(batch_size)

    def generate_trials(self, batch_size, test_mode=False):
        """
        Generate Averaging trials (2 stimuli)
        2 stimuli are presented; their average must be indicated via continuous output
        """

        # Set up trial_info 
        trial_info = self._get_trial_info(batch_size)

        for i in range(batch_size):

            # Determine sample stimulus, RFs
            sample_dirs = np.random.choice(self.n_motion_dirs, self.n_sample, replace=True)#False)
            catch       = np.random.rand() < self.catch_trial_pct

            # Generate outputs
            dif_sample_lr = (sample_dirs[0] - sample_dirs[1]) / self.n_motion_dirs * 2 * np.pi
            dif_sample_lr = dif_sample_lr % (2 * np.pi)
            resp_idx = int(dif_sample_lr // (2 * np.pi / (self.n_output - 1)) + 1)

            # Determine matchingness for this trial
            match = int(np.random.rand() < 0.5)
            if match:
                test_dir = resp_idx - 1 # -1 for looking one before fixation
            else:
                other_stim = [sample_dirs[0], # RF0 stim
                              sample_dirs[1], # RF1 stim
                              np.amin(sample_dirs), # Min
                              np.amax(sample_dirs), # Max
                              (sample_dirs[1] - sample_dirs[0]) % self.n_motion_dirs, # RF1 - RF0
                              (sample_dirs[0] + sample_dirs[1]) % self.n_motion_dirs, # Sum
                              (sample_dirs[0] + sample_dirs[1] + self.n_motion_dirs // 2) % self.n_motion_dirs] # -Sum

                test_dir = np.random.choice(np.setdiff1d(other_stim, resp_idx - 1))
                #test_dir = np.random.choice(np.setdiff1d(np.arange(self.n_motion_dirs), resp_idx - 1))

            # Establish task timings
            fix_bounds      = [0, (self.dead_time + self.fix_time)]
            rule_bounds     = [self.rule_start_time, self.rule_end_time]
            sample_bounds   = [fix_bounds[-1], fix_bounds[-1] + self.sample_time]
            delay_bounds    = [sample_bounds[-1], sample_bounds[-1] + self.delay_time]
            response_bounds = [delay_bounds[-1], delay_bounds[-1] + self.test_time]

            # Set mask at critical periods
            trial_info['train_mask'][i, response_bounds[0]:response_bounds[0]+self.mask_duration] = 0
            trial_info['train_mask'][:, range(0, self.dead_time)] = 0
            trial_info['train_mask'][i, range(*response_bounds)] *= self.test_cost_multiplier
            #trial_info['train_mask'][i, response_bounds[-1]:] = 0

            # Generate inputs
            sample_inputs = []
            for k, s_k in enumerate(sample_dirs):
                sample_inputs.append(np.reshape(self.motion_tuning[:, k, s_k],(1,-1)))
            sample_input = np.sum(np.stack(sample_inputs), axis=0)

            rule_input = int(self.n_rule_tuned > 0) * np.reshape(self.rule_tuning[:,self.rule_id],(1,-1))
            fix_input  = int(self.n_fix_tuned > 0) * np.reshape(self.fix_tuning[:,0],(1,-1))
            #cue_input  = int(self.n_cue_tuned > 0) * np.reshape(self.cue_tuning[:,0],(1,-1))
            test_input = np.reshape(self.motion_tuning[:, -1, test_dir], (1, -1))

            trial_info['neural_input'][i, range(0, response_bounds[0]), :] += fix_input 
            #trial_info['neural_input'][i, range(*response_bounds), :] += cue_input
            trial_info['neural_input'][i, range(*sample_bounds), :] += sample_input
            trial_info['neural_input'][i, range(*rule_bounds), :]   += rule_input
            trial_info['neural_input'][i, range(*response_bounds),:] += test_input

            # Until the response period, should indicate nothing
            trial_info['desired_output'][i, range(0, response_bounds[0]), 0] = 1.

            if not catch:
                trial_info['desired_output'][i, range(*response_bounds), match + 1] = 1.
            else:
                trial_info['desired_output'][i, range(0, response_bounds[0]), 0] = 1.

            # Generate reward matrix, shape (T, output_size)
            reward_matrix = np.zeros((self.trial_length, self.n_output), dtype=np.float32)
            reward_matrix[range(response_bounds[0]), 1:] = self.fix_break_penalty

            if not catch:
                reward_matrix[range(*response_bounds), 1:] = self.wrong_choice_penalty
                reward_matrix[range(*response_bounds), match + 1] = self.correct_choice_reward
                reward_matrix[-1,0] = self.fix_break_penalty
            else:
                reward_matrix[-1, 0] = self.correct_choice_reward

            trial_info['reward_matrix'][i,...] = reward_matrix

            # Record trial information
            trial_info['sample'][i,:] = sample_dirs
            trial_info['catch'][i]  = bool(catch)
            trial_info['match'][i]  = bool(match)
            trial_info['rule'][i]   = self.rule_id
            timing_dict = {'fix_bounds'     : fix_bounds,
                           'sample_bounds'  : sample_bounds,
                           'delay_bounds'   : delay_bounds,
                           'response_bounds': response_bounds}
            trial_info['timing'].append(timing_dict)


        return trial_info
