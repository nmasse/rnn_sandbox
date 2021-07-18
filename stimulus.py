import numpy as np
import tensorflow as tf

class CognitiveTasks:

    def __init__(self, args, batch_size):

        self._args    = args

        self._args.possible_tasks = 1
        self._args.n_output = 3
        self._args.batch_size = batch_size

        self.dt = self._args.dt
        self.n_input = self._args.n_input
        self.n_fix_tuned = self._args.n_fix

        self.fix_break_penalty = -1.
        self.correct_choice_reward = 1.
        self.wrong_choice_penalty = -0.01

        self.n_motion_dirs = 8
        self.n_receptive_fields = 1
        self.tuning_height = 2
        self.kappa = 2
        self.n_rules = 16
        self.n_rule_tuned = self.n_rules

        self.n_motion_tuned = self._args.n_input - self.n_rule_tuned  - self.n_fix_tuned

        self.dead_time = 100
        self.fix_time = 500
        self.sample_time = 500
        self.delay_time = 1000
        self.test_time = 500
        self.variable_delay_max = 800 // self._args.dt
        self.mask_duration = 40 // self._args.dt
        self.catch_trial_pct = 0.
        self.test_cost_multiplier = 1.
        self.rule_cue_multiplier = 1.
        self.var_delay = False

        self.trial_length = self.dead_time + self.fix_time + self.sample_time \
            + self.delay_time + self.test_time
        self.rule_onset_time = [0]
        self.rule_offset_time = [self.trial_length ]
        self.n_time_steps = self.trial_length // self._args.dt
        self.dead_time_rng = range(self.dead_time//self._args.dt)
        self.sample_time_rng = range((self.dead_time+self.fix_time)//self._args.dt, (self.dead_time+self.fix_time+self.sample_time)//self._args.dt)
        self.rule_time_rng = [range(self.rule_onset_time[n]//self._args.dt, self.rule_offset_time[n]//self._args.dt) for n in range(len(self.rule_onset_time))]
        self.input_mean = 0.
        self.noise_in = 0.1

        # generate tuning functions
        self.motion_tuning, self.fix_tuning, self.rule_tuning = self.create_tuning_functions()

        self.dataset = tf.data.Dataset.from_generator(self.generate_trial_wraper,
            args = (tf.constant(self._args.possible_tasks),),
            output_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.int8, tf.int8, tf.int8, tf.int8, tf.int8),
            output_shapes = (
                (self.n_time_steps, self._args.n_input),
                (self.n_time_steps,  self._args.n_output),
                (self.n_time_steps),
                (self.n_time_steps,  self._args.n_output),
                (), (), (), (), ()))


        #self.dataset = self.dataset.cache()
        self.dataset = self.dataset.batch(self._args.batch_size)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)


    def generate_trial_wraper(self, possible_tasks):


        while True:
            task_id = np.random.choice(possible_tasks)
            task_id = 3
            desired_output = np.zeros((self.n_time_steps,  self._args.n_output), dtype=np.float32)
            train_mask = np.ones((self.n_time_steps), dtype=np.float32)
            neural_input = np.random.normal(self.input_mean, self.noise_in, size=(self.n_time_steps, self._args.n_input))
            train_mask[self.dead_time_rng] = 0
            sample_dir = np.random.randint(self.n_motion_dirs)

            if task_id < 7:
                yield self.generate_wm_trial(task_id, desired_output, train_mask, neural_input, sample_dir)
            else:
                yield self.generate_simple_trial(task_id, desired_output, train_mask, neural_input, sample_dir)



    def generate_simple_trial(self, rule, desired_output, train_mask, neural_input, sample_dir):

        if rule == 7:
            trial_type = 'DMCgo'
            rotation_match = 0
        elif rule == 8:
            trial_type = 'DMC1go'
            rotation_match = 45
        elif rule == 9:
            trial_type = 'DMC2go'
            rotation_match = 90
        elif rule == 10:
            trial_type = 'DMCrt'
            rotation_match = 0
        elif rule == 11:
            trial_type = 'DMC1rt'
            rotation_match = 45
        elif rule == 12:
            trial_type = 'DMC2rt'
            rotation_match = 90
        elif rule == 13:
            trial_type = 'DMCdly'
            rotation_match = 0
        elif rule == 14:
            trial_type = 'DMC1dly'
            rotation_match = 45
        elif rule == 15:
            trial_type = 'DMC2dly'
            rotation_match = 90
        else:
            assert False, 'wrong rule type'

        match_rotation = int(self.n_motion_dirs*rotation_match/360)

        # Task parameters
        if 'go' in trial_type:
            stim_onset = np.random.randint(self.fix_time, self.fix_time+1000)//self.dt
            sample_time_rng = range(stim_onset, self.n_time_steps)
            fix_time_rng =  range((self.fix_time+1000)//self.dt)
            maintain_fix_time_rng = fix_time_rng
            resp_onset = fix_time_rng[-1]
            resp_time_rng =  range(resp_onset, self.n_time_steps)
        elif 'rt' in trial_type:
            stim_onset = np.random.randint(self.fix_time, self.fix_time+1000)//self.dt
            sample_time_rng = range(stim_onset, self.n_time_steps)
            fix_time_rng =  range(self.n_time_steps)
            maintain_fix_time_rng = range(stim_onset)
            resp_onset = stim_onset
            resp_time_rng = range(stim_onset, self.n_time_steps)
        elif 'dly' in trial_type:
            stim_onset = self.fix_time//self.dt
            sample_time_rng = range(stim_onset, stim_onset + 300//self.dt)
            fixation_end = stim_onset + 300//self.dt + np.random.choice([600//self.dt, 800//self.dt, 1000//self.dt])
            fix_time_rng =  range(fixation_end)
            maintain_fix_time_rng = fix_time_rng
            resp_onset = fix_time_rng[-1]
            resp_time_rng = range(fixation_end, self.n_time_steps)

        train_mask[resp_onset:resp_onset+self.mask_duration] = 0

        neural_input[sample_time_rng, :] += np.reshape(self.motion_tuning[:, 0, sample_dir],(1,-1))
        neural_input[fix_time_rng, :] += np.reshape(self.fix_tuning,(1,-1))
        neural_input[self.rule_time_rng[0], :] += np.reshape(self.rule_tuning[:,rule],(1,-1))

        sample_cat = np.floor((sample_dir+match_rotation)/(self.n_motion_dirs/2))

        desired_output[maintain_fix_time_rng,  0] = 1.
        if sample_cat == 0:
            desired_output[resp_time_rng, 1] = 1.
        else:
            desired_output[resp_time_rng, 2] = 1.

        reward_matrix = np.zeros((self.n_time_steps,  self._args.n_output), dtype=np.float32)
        reward_matrix[maintain_fix_time_rng, 1:] = self.fix_break_penalty
        if sample_cat == 0:
            reward_matrix[resp_time_rng, 1] = self.correct_choice_reward
            reward_matrix[resp_time_rng, 2] = self.wrong_choice_penalty
        else:
            reward_matrix[resp_time_rng, 1] = self.wrong_choice_penalty
            reward_matrix[resp_time_rng, 2] = self.correct_choice_reward
        reward_matrix[-1, 0] = self.fix_break_penalty


        return (neural_input, desired_output, train_mask, reward_matrix, sample_dir, -1, rule, -1, -1)




    def generate_wm_trial(self, rule, desired_output, train_mask, neural_input, sample_dir):

        if rule == 0:
            trial_type = 'DMC'
            rotation_match = 0
        elif rule == 1:
            trial_type = 'DMC1'
            rotation_match = 45
        elif rule == 2:
            trial_type = 'DMC2'
            rotation_match = 90
        elif rule == 3:
            trial_type = 'DMS'
            rotation_match = 0
        elif rule == 4:
            trial_type = 'DMRS90'
            rotation_match = 90
        elif rule == 5:
            trial_type = 'DMRS180'
            rotation_match = 180
        elif rule == 6:
            trial_type = 'DMRS270'
            rotation_match = 270

        test_RF = np.random.choice([1,2]) if trial_type == 'location_DMS' else 0

        match = np.random.randint(2)
        catch = np.random.rand() < self.catch_trial_pct

        match_rotation = int(self.n_motion_dirs*rotation_match/360)


        # Determine the delay time for this trial
        # The total trial length is kept constant, so a shorter delay implies a longer test stimulus
        if self.var_delay:
            s = int(np.random.exponential(scale=self.variable_delay_max/2))
            if s <= self.variable_delay_max:
                test_onset = (self.dead_time+self.fix_time+self.sample_time + s)//self._args.dt
            else:
                catch = 1
        else:
            test_onset = (self.dead_time+self.fix_time+self.sample_time + self.delay_time)//self._args.dt

        test_time_rng =  range(test_onset, self.n_time_steps)
        fix_time_rng =  range(test_onset)
        train_mask[test_onset:test_onset+self.mask_duration] = 0

        # Generate the sample and test stimuli based on the rule
        # DMC
        if 'DMC' in trial_type: # categorize between two equal size, contiguous zones

            possible_dirs = np.arange(self.n_motion_dirs//2)

            sample_cat = int((sample_dir)/(self.n_motion_dirs//2))
            if match == 1 and sample_cat == 0:
                test_dir = np.random.choice(possible_dirs)
            elif match == 1 and sample_cat == 1:
                test_dir = self.n_motion_dirs//2 + np.random.choice(possible_dirs)
            elif match == 0 and sample_cat == 0:
                test_dir = self.n_motion_dirs//2 + np.random.choice(possible_dirs)
            elif match == 0 and sample_cat == 1:
                test_dir = np.random.choice(possible_dirs)


            sample_dir = int((sample_dir + match_rotation) % self.n_motion_dirs)
            test_dir = int((test_dir + match_rotation)  % self.n_motion_dirs)

        # DMS or DMRS
        else:
            matching_dir = (sample_dir + match_rotation)%self.n_motion_dirs
            if match == 1: # match trial
                test_dir = matching_dir
            else:
                possible_dirs = np.setdiff1d(list(range(self.n_motion_dirs)), matching_dir)
                test_dir = possible_dirs[np.random.randint(len(possible_dirs))]

        # Calculate neural input based on sample, tests, fixation, rule, and probe
        # SAMPLE stimulus
        neural_input[self.sample_time_rng, :] += np.reshape(self.motion_tuning[:, 0, sample_dir],(1,-1))

        # TEST stimulus
        if not catch:
            neural_input[test_time_rng, :] += np.reshape(self.motion_tuning[:, test_RF, test_dir],(1,-1))

        # FIXATION cue
        if self.n_fix_tuned > 0:
            y = self.fix_tuning
            neural_input[fix_time_rng, :] += np.reshape(self.fix_tuning,(1,-1))

        # RULE CUE
        if self.n_rules> 1 and self.n_rule_tuned > 0:
            neural_input[self.rule_time_rng[0], :] += np.reshape(self.rule_tuning[:,rule],(1,-1))

        # Determine the desired network output response
        desired_output[fix_time_rng,  0] = 1.
        if not catch:
            train_mask[ test_time_rng] *= self.test_cost_multiplier # can use a greater weight for test period if needed
            if match == 0:
                desired_output[test_time_rng, 1] = 1.
            else:
                desired_output[test_time_rng, 2] = 1.
        else:
            desired_output[test_time_rng, 0] = 1.

        # Reward matrix, has shape (T, output_size)
        reward_matrix = np.zeros((self.n_time_steps,  self._args.n_output), dtype=np.float32)
        reward_matrix[fix_time_rng, 1:] = self.fix_break_penalty
        if not catch:
            if match == 0:
                reward_matrix[test_time_rng, 1] = self.correct_choice_reward
                reward_matrix[test_time_rng, 2] = self.wrong_choice_penalty
            else:
                reward_matrix[test_time_rng, 1] = self.wrong_choice_penalty
                reward_matrix[test_time_rng, 2] = self.correct_choice_reward
            reward_matrix[-1, 0] = self.fix_break_penalty
        else:
            reward_matrix[-1, 0] = self.correct_choice_reward

        return (neural_input, desired_output, train_mask, reward_matrix, sample_dir, test_dir, rule, catch, match)


    def create_tuning_functions(self, trial_type = 'DMS'):

        motion_tuning = np.zeros((self._args.n_input, self.n_receptive_fields, self.n_motion_dirs))
        fix_tuning = np.zeros((self._args.n_input))
        rule_tuning = np.zeros((self._args.n_input, self.n_rules))

        # generate list of prefered directions
        # dividing neurons by 2 since two equal groups representing two modalities
        pref_dirs = np.float32(np.arange(0,360,360/(self.n_motion_tuned//self.n_receptive_fields)))

        # generate list of possible stimulus directions
        stim_dirs = np.float32(np.arange(0,360,360/self.n_motion_dirs))

        for n in range(self.n_motion_tuned//self.n_receptive_fields):
            for i in range(self.n_motion_dirs):
                for r in range(self.n_receptive_fields):

                    if trial_type == 'distractor':
                        if n%self.n_motion_dirs == i:
                            motion_tuning[n,0,i] = self.tuning_height
                    else:
                        d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                        n_ind = n+r*self.n_motion_tuned//self.n_receptive_fields
                        motion_tuning[n_ind,r,i] = self.tuning_height*np.exp(self.kappa*d)/np.exp(self.kappa)

        for n in range(self.n_fix_tuned):
            fix_tuning[self.n_motion_tuned+n] = self.tuning_height

        for n in range(self.n_rule_tuned):
            for i in range(self.n_rules):
                if n%self.n_rules == i:
                    rule_tuning[self.n_motion_tuned+self.n_fix_tuned+n,i] = self.tuning_height*self.rule_cue_multiplier


        return motion_tuning, fix_tuning, rule_tuning
