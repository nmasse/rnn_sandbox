import numpy as np
from tasks import Task

class DMC(Task.Task):

    def __init__(self, task_name, rule_id, var_delay, dt, tuning, timing, shape, misc):

        # Initialize from superclass Task
        super().__init__(task_name, rule_id, var_delay, dt, tuning, timing, shape, misc)

    def _get_trial_info(self, batch_size):
        return super()._get_trial_info(batch_size)

    def generate_trials(self, batch_size, test_mode=False, delay_length=None):

        if self.var_delay:
            assert self.delay_max < self.test_time // 2

        trial_info = self._get_trial_info(batch_size)

        for i in range(batch_size):

            # Determine trial parameters (sample stimulus, match, etc.)
            sample_dir = np.random.randint(self.n_motion_dirs)
            sample_cat = sample_dir // (self.n_motion_dirs // 2)
            match      = np.random.randint(2)
            catch      = np.random.rand() < self.catch_trial_pct

            # Set RFs
            sample_RF = np.random.choice(self.n_RFs)
            test_RF   = np.random.choice(self.n_RFs)

            # Determine test direction based on whether it's a match trial or not
            if not test_mode:
                if match == 1: 
                    # Do not use sample_dir as a match test stimulus
                    dir0 = int(sample_cat * (self.n_motion_dirs // 2))
                    dir1 = int(self.n_motion_dirs // 2 + sample_cat * self.n_motion_dirs // 2)
                    possible_dirs = np.setdiff1d(np.arange(dir0, dir1), sample_dir)
                    test_dir = np.random.choice(possible_dirs)
                else:
                    test_dir = sample_cat*(self.n_motion_dirs//2) + np.random.randint(self.n_motion_dirs//2)
                    test_dir = np.int_((test_dir+self.n_motion_dirs//2)%self.n_motion_dirs)
            else:
                test_dir     = np.random.randint(self.n_motion_dirs)
                test_cat     = test_dir // (self.n_motion_dirs // 2)
                match        = 1 if test_cat == sample_cat else 0

            # Determine trial timing
            total_delay = self.delay_time
            total_test  = self.test_time
            if self.var_delay:
                if delay_length is not None:
                    total_delay = delay_length
                else:
                    total_delay += np.random.choice(np.arange(-self.var_delay_max, self.var_delay_max))

                total_test -= (total_delay - self.delay_time)

            fix_bounds      = [0, (self.dead_time + self.fix_time)]
            rule_bounds     = [0, self.trial_length]
            sample_bounds   = [fix_bounds[-1], fix_bounds[-1] + self.sample_time]
            delay_bounds    = [sample_bounds[-1], sample_bounds[-1] + total_delay]
            test_bounds     = [delay_bounds[-1], delay_bounds[-1] + total_test]
            response_bounds = test_bounds

            # Set mask at critical periods
            trial_info['train_mask'][i, test_bounds[0]:test_bounds[0]+self.mask_duration] = 0
            trial_info['train_mask'][:, range(0, self.dead_time)] = 0
            trial_info['train_mask'][i, range(*test_bounds)] *= self.test_cost_multiplier
            trial_info['train_mask'][i, test_bounds[-1]:] = 0

            # Generate inputs
            sample_input = np.reshape(self.motion_tuning[:, sample_RF, sample_dir],(1,-1))
            test_input   = int(catch == 0) * np.reshape(self.motion_tuning[:, test_RF, test_dir],(1,-1))
            fix_input    = int(self.n_fix_tuned > 0) * np.reshape(self.fix_tuning[:,0],(-1,1)).T
            rule_input   = int(self.n_rule_tuned > 0) * np.reshape(self.rule_tuning[:,self.rule_id],(1,-1))
            trial_info['neural_input'][i, range(*sample_bounds), :] += sample_input
            trial_info['neural_input'][i, range(*test_bounds), :]   += test_input
            trial_info['neural_input'][i, range(0, test_bounds[0]), :] += fix_input
            trial_info['neural_input'][i, range(*rule_bounds), :]   += rule_input

            # Generate outputs
            trial_info['desired_output'][i, range(0, test_bounds[0]), 0] = 1.
            if not catch:
                if match == 0:
                    trial_info['desired_output'][i, range(*test_bounds), 1] = 1. ## NON-MATCH unit
                else:
                    trial_info['desired_output'][i, range(*test_bounds), 2] = 1. ## MATCH unit
            else:
                trial_info['desired_output'][i, range(*test_bounds), 0] = 1.

            # Generate reward matrix, shape (T, output_size)
            reward_matrix = np.zeros((self.trial_length, self.n_output), dtype=np.float32)
            reward_matrix[range(response_bounds[0]), 1:] = self.fix_break_penalty
            if not catch:
                if match == 0:
                    reward_matrix[range(*response_bounds), 1:] = self.wrong_choice_penalty
                    reward_matrix[range(*response_bounds), 1] = self.correct_choice_reward
                else:
                    reward_matrix[range(*response_bounds), 1:] = self.wrong_choice_penalty
                    reward_matrix[range(*response_bounds), 2] = self.correct_choice_reward
                reward_matrix[-1, 0] = self.fix_break_penalty
            else:
                reward_matrix[-1, 0] = self.correct_choice_reward
            trial_info['reward_matrix'][i,...] = reward_matrix

            # Record trial information
            trial_info['sample'][i] = sample_dir
            trial_info['test'][i]   = test_dir
            trial_info['rule'][i]   = self.rule_id
            trial_info['catch'][i]  = catch
            trial_info['match'][i]  = match
            timing_dict = {'fix_bounds'     : fix_bounds,
                           'sample_bounds'  : sample_bounds,
                           'delay_bounds'   : delay_bounds,
                           'test_bounds'    : test_bounds,
                           'rule_bounds'    : rule_bounds,
                           'response_bounds': response_bounds}
            trial_info['timing'].append(timing_dict)

        return trial_info
