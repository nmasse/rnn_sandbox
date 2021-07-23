import numpy as np
from tasks import Task

class ABBA(Task.Task):
    # TODO: must resolve how outputs are set (e.g. in case of delay go and dms being trained in same net)
    # currently, assuming fix/match/nonmatch are 0/1/2

    def __init__(self, task_name, rule_id, var_delay, dt, tuning, timing, shape, misc):

        # Initialize from superclass Task
        super().__init__(task_name, rule_id, var_delay, dt, tuning, timing, shape, misc)

        # Store the max number of test stimuli
        self.n_tests         = misc['n_tests']
        self.match_test_prob = misc['match_test_prob']
        self.repeat_pct      = misc['repeat_pct']

    def _get_trial_info(self, batch_size):
        return super()._get_trial_info(batch_size)

    def generate_trials(self, batch_size, test_mode=False):
        """
        Generate ABBA trials
        Sample stimulis is followed by up to max_num_tests test stimuli
        Goal is to to indicate when a test stimulus matches the sample
        """

        # Set up trial_info with ABBA-specific entries, and amending the entry for test
        trial_info = self._get_trial_info(batch_size)
        trial_info['test']             = np.ones((batch_size, self.n_tests), dtype=np.float32)
        trial_info['repeat_test_stim'] = np.zeros((batch_size), dtype=np.int8)

        for i in range(batch_size):

            # Determine sample stimulus, RFs
            sample_dir = np.random.randint(self.n_motion_dirs)
            sample_RF  = np.random.choice(self.n_RFs)
            test_RFs   = np.random.choice(self.n_RFs, self.n_tests, replace=True)
            catch      = np.random.rand() < self.catch_trial_pct

            """
            Generate up to max_num_tests test stimuli
            Sequential test stimuli are identical with probability repeat_pct
            """
            stim_dirs = [sample_dir]
            test_stim_code = 0

            if test_mode:
                # Used to analyze how sample and test neuronal and synaptic tuning relate
                # not used to evaluate task accuracy
                while len(stim_dirs) <= self.n_tests:
                    q = np.random.randint(self.n_motion_dirs)
                    stim_dirs.append(q)
            else:
                while len(stim_dirs) <= self.n_tests:
                    # Handle repeated/non-repeated non-match test stimuli separately, as well
                    # as match stimuli 
                    if np.random.rand() < self.match_test_prob:
                        stim_dirs.append(sample_dir)
                    else:
                        if len(stim_dirs) > 1  and np.random.rand() < self.repeat_pct:
                            # Repeat last stimulus
                            stim_dirs.append(stim_dirs[-1])
                            trial_info['repeat_test_stim'][i] = 1
                        else:
                            possible_dirs = np.setdiff1d(np.arange(self.n_motion_dirs), [stim_dirs])
                            distractor_dir = possible_dirs[np.random.randint(len(possible_dirs))]
                            stim_dirs.append(distractor_dir)

            # Establish task timings
            fix_bounds      = [0, (self.dead_time + self.fix_time)]
            rule_bounds     = [0, self.trial_length]
            sample_bounds   = [fix_bounds[-1], fix_bounds[-1] + self.sample_time]

            # Timings with multiple elements: delay and test
            delay_bounds    = [sample_bounds[-1], sample_bounds[-1] + self.delay_time]
            test_bounds     = [delay_bounds[-1], delay_bounds[-1] + self.test_time]
            for _ in range(1, self.n_tests):
                delay_bounds.append([test_bounds[-1][1], test_bounds[-1][1] + self.delay_time])
                test_bounds.append([delay_bounds[-1][1], delay_bounds[-1][1] + self.test_time])
            response_bounds = test_bounds

            # Set mask at critical periods
            for tb in test_bounds:
                trial_info['train_mask'][tb[0]:tb[0] + mask_duration, i] = 0
                trial_info['train_mask'][range(*tb), i] *= self.test_cost_multiplier
            trial_info['train_mask'][range(0, self.dead_time), :] = 0
            

            # Generate inputs
            sample_input = np.reshape(self.motion_tuning[:, sample_RF, sample_dir],(1,-1))
            test_inputs  = []
            for k in range(len(stim_dirs) - 1):
                test_inp = np.reshape(self.motion_tuning[:, test_RFs[k], 
                    stim_dirs[k + 1]],(1,-1))
                test_inputs.append(int(catch != 0) * test_inp)
            fix_input    = int(self.num_fix_tuned > 0) * np.reshape(self.fix_tuning[:,0],(-1,1)).T
            rule_input   = int(self.num_rule_tuned > 0) * np.reshape(self.rule_tuning[:,self.rule_id],(1,-1))
            trial_info['neural_input'][range(*sample_bounds), i, :] += sample_input
            trial_info['neural_input'][range(*rule_bounds), i, :]   += rule_input
            trial_info['neural_input'][range(0, delay_bounds[0][0]), i] += fix_input
            for db, tb, ti in zip(delay_bounds, test_bounds, test_inputs):
                trial_info['neural_input'][range(*db), i] += fix_input
                trial_info['neural_input'][range(*tb), i, :] += ti

            # Generate outputs
            trial_info['desired_output'][range(0, test_bounds[0][0]), i, 0] = 1.
            if not catch:
                for db, tb, test_dir in zip(delay_bounds, test_bounds, stim_dirs[1:]):
                    trial_info['desired_output'][range(*db), i, 0] = 1.
                    if test_dir != sample_dir:
                        trial_info['desired_output'][range(*tb), i, 1] = 1. ## NON-MATCH unit
                    else:
                        trial_info['desired_output'][range(*tb), i, 2] = 1. ## MATCH unit
            else:
                for db, tb in zip(delay_bounds, test_bounds):
                    trial_info['desired_output'][range(*db), i, 0] = 1.
                    trial_info['desired_output'][range(*tb), i, 0] = 1.

            # Generate reward matrix, shape (T, output_size)
            reward_matrix = np.zeros((self.trial_length, self.n_output), dtype=np.float32)
            if not catch:
                reward_matrix[range(0, test_bounds[0][0]), i, 1:] = self.fix_break_penalty
                for db, tb, test_dir in zip(delay_bounds, test_bounds, stim_dirs[1:]):
                    reward_matrix[range(*db), i, 1:] = self.fix_break_penalty
                    if test_dir != sample_dir:
                        reward_matrix[range(*tb), i, 1] = self.correct_choice_reward ## NON-MATCH unit
                        reward_matrix[range(*tb), i, 2:] = self.wrong_choice_penalty
                    else:
                        reward_matrix[range(*tb), i, 1:] = self.wrong_choice_penalty ## MATCH unit
                        reward_matrix[range(*tb), i, 2] = self.correct_choice_reward
                reward_matrix[-1, 0] = self.fix_break_penalty
            else:
                reward_matrix[-1, 0] = self.correct_choice_reward
            trial_info['reward_matrix'][i,...] = reward_matrix

            # Record trial information
            trial_info['sample'][i] = sample_dir
            trial_info['test'][i]   = np.int8(stim_dirs)
            trial_info['rule'][i]   = rule_id
            trial_info['catch'][i]  = catch
            timing_dict = {'fix_bounds'     : fix_bounds,
                           'sample_bounds'  : sample_bounds,
                           'delay_bounds'   : delay_bounds,
                           'test_bounds'    : test_bounds,
                           'rule_bounds'    : rule_bounds,
                           'response_bounds': response_bounds}
            trial_info['timing'].append(timing_dict)


        return trial_info
