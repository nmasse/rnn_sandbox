import numpy as np
from tasks import Task

class DelayGo(Task.Task):

    def __init__(self, task_name, rule_id, var_delay, dt, tuning, timing, shape, misc):

        # Initialize from superclass Task
        super().__init__(task_name, rule_id, var_delay, dt, tuning, timing, shape, misc)
        self.categorization = misc['categorization']

    def _get_trial_info(self, batch_size):
        return super()._get_trial_info(batch_size)

    def generate_trials(self, batch_size, test_mode=False, delay_length=None):

        if self.var_delay:
            assert self.delay_max < self.test_time // 2

        trial_info = self._get_trial_info(batch_size)

        for i in range(batch_size):

            # Determine trial parameters (sample stimulus, match, etc.)
            sample_dir = np.random.randint(self.n_motion_dirs)
            catch      = np.random.rand() < self.catch_trial_pct

            # Set RFs
            sample_RF = np.random.choice(self.n_RFs)

            # Determine trial timing
            total_delay = self.delay_time
            total_test  = self.test_time
            if self.var_delay:
                if delay_length is not None:
                    total_delay = delay_length
                else:
                    total_delay += np.random.choice(np.arange(-self.var_delay_max, self.var_delay_max))

                total_test -= (total_delay - self.delay_time)

            # Establish task epoch bounds (test period = go period in this task)
            fix_bounds      = [0, (self.dead_time + self.fix_time)]
            rule_bounds     = [0, self.trial_length]
            sample_bounds   = [fix_bounds[-1], fix_bounds[-1] + self.sample_time]
            delay_bounds    = [sample_bounds[-1], sample_bounds[-1] + total_delay]
            test_bounds     = [delay_bounds[-1], delay_bounds[-1] + total_test]
            response_bounds = test_bounds
            cue_bounds      = test_bounds # When GO cue is to be presented

            # Set mask at critical periods
            trial_info['train_mask'][i, test_bounds[0]:test_bounds[0]+self.mask_duration] = 0
            trial_info['train_mask'][:, range(0, self.dead_time)] = 0
            trial_info['train_mask'][i, range(*test_bounds)] *= self.test_cost_multiplier
            trial_info['train_mask'][i, test_bounds[-1]:] = 0

            # Generate inputs
            sample_input = np.reshape(self.motion_tuning[:, sample_RF, sample_dir],(1,-1))
            fix_input    = int(self.n_fix_tuned > 0) * np.reshape(self.fix_tuning[:,0],(1,-1))
            rule_input   = int(self.n_rule_tuned > 0) * np.reshape(self.rule_tuning[:,self.rule_id],(1,-1))
            #cue_input    = int(self.n_cue_tuned > 0) * np.reshape(self.cue_tuning[:,0],(1,-1))
            trial_info['neural_input'][i, range(*sample_bounds), :] += sample_input
            trial_info['neural_input'][i, range(0, test_bounds[0]), :] += fix_input
            trial_info['neural_input'][i, range(*rule_bounds), :]   += rule_input
            #trial_info['neural_input'][i, range(*cue_bounds), :]    += cue_input

            if self.categorization:
                N = self.n_motion_dirs // 2
                resp_idx = 1 + sample_dir // N
            else:
                resp_idx = 1 + sample_dir

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
            trial_info['sample'][i,0] = sample_dir
            trial_info['catch'][i] = bool(catch)
            trial_info['rule'][i] = self.rule_id
            timing_dict = {'fix_bounds'     : fix_bounds,
                           'sample_bounds'  : sample_bounds,
                           'delay_bounds'   : delay_bounds,
                           'test_bounds'    : test_bounds,
                           'cue_bounds'     : cue_bounds,
                           'response_bounds': response_bounds}
            trial_info['timing'].append(timing_dict)

        return trial_info
