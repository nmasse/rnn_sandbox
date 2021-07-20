import numpy as np
from tasks import Task

class DualDMS(Task.Task):

    def __init__(self, task_name, rule_id, var_delay, dt, tuning, timing, shape, misc):

        # Initialize from superclass Task
        super().__init__(task_name, rule_id, var_delay, dt, tuning, timing, shape, misc)

    def _get_trial_info(self, batch_size):
        return super()._get_trial_info(batch_size)

    def generate_dualDMS_trial(self, test_mode):

        """
        Generate a trial based on "Reactivation of latent working memories with transcranial magnetic stimulation"

        Trial outline
        1. Dead period
        2. Fixation
        3. Two sample stimuli presented
        4. Delay (cue in middle, and possibly probe later)
        5. Test stimulus (to cued modality, match or non-match)
        6. Delay (cue in middle, and possibly probe later)
        7. Test stimulus

        INPUTS:
        1. sample_time (duration of sample stimlulus)
        2. test_time
        3. delay_time
        4. cue_time (duration of rule cue, always presented halfway during delay)
        5. probe_time (usually set to one time step, always presented 3/4 through delay)
        """

        test_time_rng = []
        mask_time_rng = []

        for n in range(2):
            test_time_rng.append(range((par['dead_time']+par['fix_time']+par['sample_time']+(n+1)*par['delay_time']+n*par['test_time'])//par['dt'], \
                (par['dead_time']+par['fix_time']+par['sample_time']+(n+1)*par['delay_time']+(n+1)*par['test_time'])//par['dt']))
            mask_time_rng.append(range((par['dead_time']+par['fix_time']+par['sample_time']+(n+1)*par['delay_time']+n*par['test_time'])//par['dt'], \
                (par['dead_time']+par['fix_time']+par['sample_time']+(n+1)*par['delay_time']+n*par['test_time']+par['mask_duration'])//par['dt']))


        fix_time_rng = []
        fix_time_rng.append(range(par['dead_time']//par['dt'], (par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time'])//par['dt']))
        fix_time_rng.append(range((par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time'])//par['dt'], \
            (par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+par['test_time'])//par['dt']))


        # duration of mask after test onset
        mask_duration = par['mask_duration']//par['dt']


        trial_info = {'desired_output'  :  np.zeros((par['num_time_steps'], par['batch_size'], par['n_output']),dtype=np.float32),
                      'train_mask'      :  np.ones((par['num_time_steps'], par['batch_size']),dtype=np.float32),
                      'sample'          :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'test'            :  np.zeros((par['batch_size'],2,2),dtype=np.int8),
                      'test_mod'        :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'rule'            :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'match'           :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'catch'           :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'probe'           :  np.zeros((par['batch_size'],2),dtype=np.int8),
                      'neural_input'    :  np.random.normal(par['input_mean'], par['noise_in'], size=(par['num_time_steps'], par['batch_size'], par['n_input']))}


        for t in range(par['batch_size']):

            # generate sample, match, rule and prob params
            for i in range(2):
                trial_info['sample'][t,i] = np.random.randint(par['num_motion_dirs'])
                trial_info['match'][t,i] = np.random.randint(2)
                trial_info['rule'][t,i] = np.random.randint(2)
                trial_info['catch'][t,i] = np.random.rand() < par['catch_trial_pct']
                if i == 1:
                    # only generate a pulse during 2nd delay epoch
                    trial_info['probe'][t,i] = np.random.rand() < par['probe_trial_pct']


            # determine test stimulu based on sample and match status
            for i in range(2):

                if test_mode:
                    trial_info['test'][t,i,0] = np.random.randint(par['num_motion_dirs'])
                    trial_info['test'][t,i,1] = np.random.randint(par['num_motion_dirs'])
                else:
                    # if trial is not a catch, the upcoming test modality (what the network should be attending to)
                    # is given by the rule cue
                    if not trial_info['catch'][t,i]:
                        trial_info['test_mod'][t,i] = trial_info['rule'][t,i]
                    else:
                        trial_info['test_mod'][t,i] = (trial_info['rule'][t,i]+1)%2

                    # cued test stimulus
                    if trial_info['match'][t,i] == 1:
                        trial_info['test'][t,i,0] = trial_info['sample'][t,trial_info['test_mod'][t,i]]
                    else:
                        sample = trial_info['sample'][t,trial_info['test_mod'][t,i]]
                        bad_directions = [sample]
                        possible_stim = np.setdiff1d(list(range(par['num_motion_dirs'])), bad_directions)
                        trial_info['test'][t,i,0] = possible_stim[np.random.randint(len(possible_stim))]

                    # non-cued test stimulus
                    trial_info['test'][t,i,1] = np.random.randint(par['num_motion_dirs'])


            """
            Calculate input neural activity based on trial params
            """
            # SAMPLE stimuli
            trial_info['neural_input'][par['sample_time_rng'], t, :] += np.reshape(self.motion_tuning[:,0,trial_info['sample'][t,0]],(1,-1))
            trial_info['neural_input'][par['sample_time_rng'], t, :] += np.reshape(self.motion_tuning[:,1,trial_info['sample'][t,1]],(1,-1))

            # Cued TEST stimuli
            trial_info['neural_input'][test_time_rng[0], t, :] += np.reshape(self.motion_tuning[:,trial_info['test_mod'][t,0],trial_info['test'][t,0,0]],(1,-1))
            trial_info['neural_input'][test_time_rng[1], t, :] += np.reshape(self.motion_tuning[:,trial_info['test_mod'][t,1],trial_info['test'][t,1,0]],(1,-1))

            # Non-cued TEST stimuli
            trial_info['neural_input'][test_time_rng[0], t, :] += np.reshape(self.motion_tuning[:,(1+trial_info['test_mod'][t,0])%2,trial_info['test'][t,0,1]],(1,-1))
            trial_info['neural_input'][test_time_rng[1], t, :] += np.reshape(self.motion_tuning[:,(1+trial_info['test_mod'][t,1])%2,trial_info['test'][t,1,1]],(1,-1))


            # FIXATION
            trial_info['neural_input'][fix_time_rng[0], t, :] += np.reshape(self.fix_tuning[:,0],(1,-1))
            trial_info['neural_input'][fix_time_rng[1], t, :] += np.reshape(self.fix_tuning[:,0],(1,-1))

            # RULE CUE
            trial_info['neural_input'][par['rule_time_rng'][0], t, :] += np.reshape(self.rule_tuning[:,trial_info['rule'][t,0]],(1,-1))
            trial_info['neural_input'][par['rule_time_rng'][1], t, :] += np.reshape(self.rule_tuning[:,trial_info['rule'][t,1]],(1,-1))

            # PROBE
            # increase reponse of all stim tuned neurons by 10
            """
            if trial_info['probe'][t,0]:
                trial_info['neural_input'][:est,probe_time1,t] += 10
            if trial_info['probe'][t,1]:
                trial_info['neural_input'][:est,probe_time2,t] += 10
            """

            """
            Desired outputs
            """
            # FIXATION
            trial_info['desired_output'][fix_time_rng[0], t, 0] = 1
            trial_info['desired_output'][fix_time_rng[1], t, 0] = 1
            # TEST 1
            trial_info['train_mask'][ test_time_rng[0], t] *= par['test_cost_multiplier'] # can use a greater weight for test period if needed
            if trial_info['match'][t,0] == 1:
                trial_info['desired_output'][test_time_rng[0], t, 2] = 1
            else:
                trial_info['desired_output'][test_time_rng[0], t, 1] = 1
            # TEST 2
            trial_info['train_mask'][ test_time_rng[1], t] *= par['test_cost_multiplier'] # can use a greater weight for test period if needed
            if trial_info['match'][t,1] == 1:
                trial_info['desired_output'][test_time_rng[1], t, 2] = 1
            else:
                trial_info['desired_output'][test_time_rng[1], t, 1] = 1

            # set to mask equal to zero during the dead time, and during the first times of test stimuli
            trial_info['train_mask'][:par['dead_time']//par['dt'], t] = 0
            trial_info['train_mask'][mask_time_rng[0], t] = 0
            trial_info['train_mask'][mask_time_rng[1], t] = 0

        return trial_info

    def generate_trials(self, batch_size, test_mode=False, delay_length=None):

        if self.var_delay:
            assert self.delay_max < self.test_time // 2

        trial_info = self._get_trial_info()

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
