import numpy as np
from tasks import Task

class ABBA(Task.Task):
    # TODO: must resolve how outputs are set (e.g. in case of delay go and dms being trained in same net)
    # currently, assuming fix/match/nonmatch are 0/1/2

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
            sample_dir = np.random.randint(self.num_motion_dirs)
            match      = np.random.randint(2)
            catch      = np.random.rand() < self.catch_trial_pct
            match_rotation = int(self.num_motion_dirs * self.rotation/360)

            # Set RFs
            sample_RF = np.random.choice(self.n_RFs)
            test_RF   = np.random.choice(self.n_RFs)

            # Determine test direction based on whether it's a match trial or not
            if not test_mode:
                matching_dir = (sample_dir + match_rotation) % self.num_motion_dirs
                if match == 1: 
                    test_dir = matching_dir
                else:
                    possible_dirs = np.setdiff1d(np.arange(self.num_motion_dirs, dtype=np.int32), matching_dir)
                    test_dir      = np.random.choice(possible_dirs)
            else:
                test_dir     = np.random.randint(self.num_motion_dirs)
                matching_dir = (sample_dir + match_rotation) % self.num_motion_dirs
                match        = 1 if test_dir == matching_dir else 0

            # Determine trial timing
            total_delay = self.delay_time
            total_test  = self.test_time
            if self.var_delay:
                if delay_length is not None:
                    total_delay = delay_length
                else:
                    total_delay += np.random.choice(np.arange(-self.var_delay_max, var_var_delay_max))

                total_test -= (total_delay - self.delay_time)

            fix_bounds      = [0, (self.dead_time + self.fix_time)]
            rule_bounds     = [fix_bounds[-1] - self.rule_time, fix_bounds[-1]]
            sample_bounds   = [fix_bounds[-1], fix_bounds[-1] + self.sample_time]
            delay_bounds    = [sample_bounds[-1], sample_bounds[-1] + total_delay]
            test_bounds     = [delay_bounds[-1], delay_bounds[-1] + total_test]
            response_bounds = test_bounds

            # Set mask at critical periods
            trial_info['train_mask'][test_bounds[0]:test_bounds[0]+mask_duration, i] = 0
            trial_info['train_mask'][range(0, self.dead_time), :] = 0
            trial_info['train_mask'][range(*test_bounds), i] *= self.test_cost_multiplier

            # Generate inputs
            sample_input = np.reshape(self.motion_tuning[:, sample_RF, sample_dir],(1,-1))
            test_input   = int(catch != 0) * np.reshape(self.motion_tuning[:, test_RF, test_dir],(1,-1))
            fix_input    = int(self.num_fix_tuned > 0) * np.reshape(self.fix_tuning[:,0],(-1,1)).T
            rule_input   = int(self.num_rule_tuned > 0) * np.reshape(self.rule_tuning[:,self.rule_id],(1,-1))
            trial_info['neural_input'][range(*sample_bounds), i, :] += sample_input
            trial_info['neural_input'][range(*test_bounds), i, :]   += test_input
            trial_info['neural_input'][range(0, test_bounds[0]), i] += fix_input
            trial_info['neural_input'][range(*rule_bounds), i, :]   += rule_input

            # Generate outputs
            trial_info['desired_output'][range(0, test_bounds[0]), i, 0] = 1.
            if not catch:
                if match == 0:
                    trial_info['desired_output'][range(*test_bounds), i, 1] = 1. ## NON-MATCH unit
                else:
                    trial_info['desired_output'][range(*test_bounds), i, 2] = 1. ## MATCH unit
            else:
                trial_info['desired_output'][range(*test_bounds), i, 0] = 1.

            # Record trial information
            trial_info['sample'][i] = sample_dir
            trial_info['test'][i]   = test_dir
            trial_info['rule'][i]   = rule
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
