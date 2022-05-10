import numpy as np
import gym
import copy
from . import * 
import tensorflow as tf
import matplotlib.pyplot as plt


class TaskGym(gym.Env):

    def __init__(self, task_list, batch_size, rnn_params, 
        buffer_size=20000, new_task_prob=1.):

        self.n_tasks = len(task_list)
        self.new_task_prob = new_task_prob
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.task_manager = TaskManager(
                            task_list,
                            buffer_size,
                            n_motion_tuned=rnn_params.n_motion_tuned,
                            n_fix_tuned=rnn_params.n_fix_tuned)
        self.n_bottom_up = rnn_params.n_bottom_up
        self.n_top_down = rnn_params.n_top_down
        self.non_motion_mult = self.task_manager.non_motion_mult
        self.trials_per_task = [self.task_manager.generate_batch(buffer_size, rule=n) 
            for n in range(self.n_tasks)]
        self.task_id = np.random.choice(self.n_tasks, size = (batch_size))
        self.trial_id = np.random.choice(buffer_size, size = (batch_size))
        self.time = np.zeros((batch_size), dtype=np.int32)
        self.clipob = 10.
        self.cliprew = 10.

    def analyze_trials_per_task(self):

        # 1. Confirm that # match and # non match trials are about equal
        # for all tasks


        # 2. For each stimulus trio, that is, each pair of samples and test,
        # compute across all tasks the overall number of times that grouping
        # was paired w/ a match vs. a non-match output (should be roughly 
        # equal)


        return


    def split_observation(self, observation):
        # split neural input into bottom-up (motion direction activity) for RNN
        # and top-down (context/rule) for Cont Actor RL
        bottom_up = observation[:, :self.n_bottom_up]
        top_down = observation[:, -self.n_top_down:] / self.non_motion_mult
        return bottom_up, top_down

    def level_up(self):
        self.curriculum = np.minimum(self.curriculum+1, self.n_tasks)
        return

    def reset_all(self):

        observations = []
        masks = []
        for i in range(self.batch_size):
            obs, mask = self.reset(i)
            observations.append(obs)
            masks.append(mask)

        observations = np.stack(observations)
        masks = np.stack(masks)
        bottom_up, top_down = self.split_observation(observations)
        return bottom_up, top_down, masks


    def reset(self, agent_id):

        self.time[agent_id] = 0
        new_task = np.random.choice([True, False], p=[self.new_task_prob, 1-self.new_task_prob])
        if new_task:
            self.task_id[agent_id] = np.random.choice(self.n_tasks)
        self.trial_id[agent_id] = np.random.choice(self.buffer_size)
        obs = self.trials_per_task[self.task_id[agent_id]][0][self.trial_id[agent_id], self.time[agent_id], :]
        mask = self.trials_per_task[self.task_id[agent_id]][2][self.trial_id[agent_id], self.time[agent_id]]

        return obs, mask

    def step_all(self, actions):

        rewards = []
        dones = []
        observations = []
        masks = []
        for i in range(self.batch_size):
            observation, mask, reward, done = self.step(i, actions[i])
            observations.append(observation)
            masks.append(mask)
            rewards.append(reward)
            dones.append(done)

        bottom_up, top_down = self.split_observation(np.stack(observations))

        return bottom_up, top_down, np.stack(masks), np.stack(rewards), np.stack(dones)

    def step(self, agent_id, action):

        trial = self.trial_id[agent_id]
        task  = self.task_id[agent_id]
        time  = self.time[agent_id]

        reward = self.trials_per_task[task][3][trial, time, action]
        mask   = self.trials_per_task[task][2][trial, time]

        reward *= mask

        if reward != 0:
            done = True
            observation, mask = self.reset(agent_id)
        else:
            done = False
            self.time[agent_id] += 1
            time = self.time[agent_id]
            observation = self.trials_per_task[task][0][trial, time, :]
            mask = self.trials_per_task[task][2][trial, time]

        return observation, mask, reward, done

class TaskManager:

    def __init__(self,
                 task_list,
                 batch_size,
                 n_motion_tuned = 32,
                 n_fix_tuned = 1,
                 tuning_height = 2,
                 kappa = 2,
                 non_motion_mult = 2.,
                 dt = 20,
                 input_mean = 0,
                 input_noise = 0.0,
                 catch_trial_pct = 0.,
                 test_cost_mult = 1.,
                 fix_break_penalty = -1.0,
                 correct_choice_reward = 1.0,
                 wrong_choice_penalty = -0.01, 
                 tf2 = False,
                 random_seed=0):
        ## Args:
        # task_list: list of dicts of task IDs w/ task parameters
        #   - n directions
        #   - n cues
        #   - n RFs
        #   - var_delay
        #   - timing dictionary
        self.n_motion_tuned  = n_motion_tuned
        self.n_fix_tuned     = n_fix_tuned
        self.tuning_height   = tuning_height
        self.kappa           = kappa
        self.non_motion_mult = non_motion_mult
        self.dt              = dt
        self.input_mean      = input_mean
        self.input_noise     = input_noise
        self.catch_trial_pct = catch_trial_pct
        self.test_cost_mult  = test_cost_mult
        self.batch_size      = batch_size
        self.tf2             = tf2

        # Bind RL information
        self.fix_break_penalty     = fix_break_penalty
        self.correct_choice_reward = correct_choice_reward
        self.wrong_choice_penalty  = wrong_choice_penalty

        # Set random seed
        np.random.seed(random_seed)

        # Filter for DMS, AntiDMS, ABCA among task_list specified
        for t in task_list:
            if t['name'] == 'DMS':
                t['name'] = 'DMRS'
                t['rotation'] = 0
            elif t['name'] == 'AntiDMS':
                t['name'] = 'DMRS'
                t['rotation'] = 180

            elif t['name'] == 'ABCA':
                t['name'] = 'ABBA'

        # Use number of task_list to determine number of rule-tuned inputs,
        # cue-tuned inputs
        self.n_rule_tuned = len(task_list)
        self.n_cue_tuned  = max([t['n_cues'] for t in task_list])

        # Determine number of RFs required, num motion directions; duplicate
        # motion tuning for each receptive field
        self.n_RFs = max([t['n_RFs'] for t in task_list])
        self.n_motion_dirs = max([t['n_motion_dirs'] for t in task_list])
        self.n_motion_tuned *= self.n_RFs

        # Determine maximum number of sample/test stimuli
        self.n_sample = max([t['n_sample'] for t in task_list])
        self.n_test   = max([t['n_test'] for t in task_list])

        self.n_input = self.n_motion_tuned + \
                       self.n_rule_tuned   + \
                       self.n_cue_tuned    + \
                       self.n_fix_tuned
        self.n_output = np.sum(np.unique([t['n_output'] for t in task_list]))


        # Build tuning, shape
        self.tuning = self.create_tuning_functions()
        self.shape  = self.get_shape()

        # Build and bind task objects
        self.task_list, self.task_names = [], []
        for i, t in enumerate(task_list):
            self.task_names.append(t['name'])
            task = eval(f"{t['name']}.{t['name']}")

            # Create the task-specific dictionary of miscellaneous params
            misc = copy.copy(t)
            misc['input_mean']            = self.input_mean
            misc['input_noise']           = self.input_noise
            misc['catch_trial_pct']       = self.catch_trial_pct
            misc['test_cost_multiplier']  = self.test_cost_mult
            misc['fix_break_penalty']     = self.fix_break_penalty
            misc['correct_choice_reward'] = self.correct_choice_reward
            misc['wrong_choice_penalty']  = self.wrong_choice_penalty
            misc['n_sample']              = self.n_sample
            misc['n_test']                = self.n_test
            #misc['n_RFs']                 = self.n_RFs # make this equal to max number of RFs

            task_args = [t['name'], i, t['var_delay'], self.dt, self.tuning, t['timing'], self.shape, misc]
            self.task_list.append(task(*task_args))

        # Append dataset object for tf2 interface
        if self.tf2:
            self.dataset = tf.data.Dataset.from_generator(self.generate_batch_tf2,
                output_types = (tf.float32, tf.float32, tf.float32, tf.float32,
                    tf.int8, tf.int8),
                output_shapes = (
                    (self.trial_length, self.n_input), # neural_input
                    (self.trial_length, self.n_output), # desired_output
                    (self.trial_length), # train_mask
                    (self.trial_length, self.n_output), # reward_matrix
                    (), # sample_dir
                    ())) # rule

            self.dataset = self.dataset.batch(self.batch_size)
            self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def visualize_task(self, rule_id):
        # Visualize neural inputs and desired outputs of specified task
        trial_info = self.generate_batch(1, rule=rule_id)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
        ax[0].imshow(trial_info[0].squeeze().T, aspect='equal')
        ax[1].imshow(trial_info[1].squeeze().T, aspect='equal')
        plt.show()


    def generate_batch_tf2(self):
        while True:
            yield self.generate_batch(1)

    def generate_batch(self, batch_size, to_exclude=[], rule=None, include_test=False, 
        weights=None, **kwargs):
        # Generate a batch of trials; if rule is specified,
        # only generate trials of that rule, otherwise
        # generate at random from all task_list interleaved
        # w/in same batch
        if rule is not None:
            trial_info = self.task_list[rule].generate_trials(batch_size, **kwargs)
        else:
            # Get empty trial info dictionary, then write its elements,
            # one at a time
            trial_info = self.generate_empty_trial_info(batch_size, to_exclude, include_test)
            for i in range(batch_size):
                rule = np.random.choice(np.setdiff1d(np.arange(self.n_rule_tuned), to_exclude),
                    p=weights)
                trial_info = self.write_trial(trial_info,
                    self.task_list[rule].generate_trials(1, **kwargs),
                    i,
                    batch_size)

        # Extract just the elements that are necessary for the tf dataset
        keys = ['neural_input', 'desired_output', 'train_mask',
            'reward_matrix', 'sample', 'rule']
        if include_test:
            keys.append('test')

        return tuple([trial_info[k].squeeze().astype(np.float32) for k in keys])


    def generate_empty_trial_info(self, batch_size, to_exclude, include_test=False):
        batch_trial_length = max([t.trial_length for j, t in enumerate(self.task_list)
            if j not in to_exclude])

        # Generate an empty trial info dictionary to add other trials to, one-by-one
        trial_info = {'desired_output'  :  np.zeros((batch_size, batch_trial_length, self.n_output), dtype=np.float32),
                      'train_mask'      :  np.ones((batch_size, batch_trial_length), dtype=np.float32),
                      'rule'            :  np.zeros((batch_size), dtype=np.int8),
                      'neural_input'    :  np.random.normal(self.input_mean, self.input_noise,
                                                size=(batch_size, batch_trial_length, self.n_input)),
                      'sample'          :  -np.ones((batch_size, self.n_sample), dtype=np.int8),
                      'reward_matrix'   :  np.zeros((batch_size, batch_trial_length, self.n_output), dtype=np.float32),
                      'retrospective'   :  np.full((batch_size), False),
                      'timing'          :  [],
                      'task_specific'   :  []}
        if include_test:
            trial_info['test'] = -np.ones((batch_size, self.n_test), dtype=np.int8)

        return trial_info

    def write_trial(self, trial_info, cur_trial, i, batch_size):
        # Write trial information to existing trial_info dictionary
        task_specific = {k: v for k, v in cur_trial.items() if k not in trial_info.keys()}
        for k, v in cur_trial.items():
            if k == 'timing':
                trial_info['timing'].append(v)
            elif k in trial_info.keys():
                trial_info[k][i] = v
        trial_info['task_specific'].append(task_specific)

        return trial_info

    def get_shape(self):
        # Package together relevant shapes for setting task_list up
        shape = {}
        shape['n_input']        = self.n_input
        shape['n_output']       = self.n_output
        shape['n_motion_tuned'] = self.n_motion_tuned
        shape['n_fix_tuned']    = self.n_fix_tuned
        shape['n_rule_tuned']   = self.n_rule_tuned
        shape['n_cue_tuned']    = self.n_cue_tuned

        return shape

    def create_tuning_functions(self):

        """
        Generate motion direction-tuned input units
        """

        motion_tuning = np.zeros((self.n_input, self.n_RFs, self.n_motion_dirs))
        fix_tuning    = np.zeros((self.n_input, self.n_fix_tuned))
        rule_tuning   = np.zeros((self.n_input, self.n_rule_tuned))
        cue_tuning    = np.zeros((self.n_input, self.n_cue_tuned))

        # Generate list of prefered directions, dividing neurons by n_RFs
        pref_dirs = np.float32(np.arange(0, 360, 360 / (self.n_motion_tuned //
            self.n_RFs)))

        # Generate list of possible stimulus directions
        stim_dirs = np.float32(np.arange(0, 360, 360 / self.n_motion_dirs))

        ###
        # Generate motion tuning
        for n in range(self.n_motion_tuned // self.n_RFs):
            for i in range(self.n_motion_dirs):
                for r in range(self.n_RFs):
                    d = np.cos((stim_dirs[i] - pref_dirs[n])/180*np.pi)
                    n_ind = n+r*self.n_motion_tuned//self.n_RFs
                    motion_tuning[n_ind,r,i] = self.tuning_height*np.exp(self.kappa*d)/np.exp(self.kappa)

        ###
        # Generate fixation tuning
        for n in range(self.n_fix_tuned):
            fix_tuning[self.n_motion_tuned + n, 0] = self.non_motion_mult * self.tuning_height

        ###
        # Generate cue tuning
        for n in range(self.n_cue_tuned):
            start = self.n_motion_tuned + self.n_fix_tuned
            cue_tuning[start + n, n] = self.non_motion_mult * self.tuning_height

        ###
        # Generate rule tuning
        for n in range(self.n_rule_tuned):
            start = self.n_motion_tuned + self.n_fix_tuned + self.n_cue_tuned
            rule_tuning[start+n,n] = self.non_motion_mult * self.tuning_height

        # Package together into a dictionary and return
        tuning = {'motion': motion_tuning,
                  'fix'   : fix_tuning,
                  'rule'  : rule_tuning,
                  'cue'   : cue_tuning}
        return tuning

def twostim_matchingtasks():

    generic_timing = {'dead_time'   : 300,
                      'fix_time'    : 200,
                      'sample_time'  : 300,
                      'delay_time'   : 1000,
                      'test_time'    : 300}
    generic_hps = {'n_motion_dirs': 8,
                   'n_sample': 2,
                   'n_RFs': 3,
                   'n_cues': 0, # No cues
                   'var_delay': False,
                   'distractor': False,
                   'n_output': 3,
                   'n_test': 1,
                   'mask_duration': 40,
                   'var_delay_max': 200,
                   'trial_length': sum(generic_timing.values()),
                   'timing': generic_timing}

    # All 2-stim tasks share the same set of meta-params
    AveragingMatching = generic_hps.copy()
    AveragingMatching['name'] = 'TwoStimAveragingMatching'

    AveragingOppositeMatching = generic_hps.copy()
    AveragingOppositeMatching['name'] = 'TwoStimAveragingOppositeMatching'

    LeftIndicationMatching = generic_hps.copy()
    LeftIndicationMatching['name'] = 'TwoStimLeftIndicationMatching'

    LeftIndicationOppositeMatching = generic_hps.copy()
    LeftIndicationOppositeMatching['name'] = 'TwoStimLeftIndicationOppositeMatching'

    RightIndicationMatching = generic_hps.copy()
    RightIndicationMatching['name'] = 'TwoStimRightIndicationMatching'

    RightIndicationOppositeMatching = generic_hps.copy()
    RightIndicationOppositeMatching['name'] = 'TwoStimRightIndicationOppositeMatching'

    MinIndicationMatching = generic_hps.copy()
    MinIndicationMatching['name'] = 'TwoStimMinIndicationMatching'

    MaxIndicationMatching = generic_hps.copy()
    MaxIndicationMatching['name'] = 'TwoStimMaxIndicationMatching'

    SubtractingLeftRightMatching = generic_hps.copy()
    SubtractingLeftRightMatching['name'] = 'TwoStimSubtractingLeftRightMatching'

    SubtractingRightLeftMatching = generic_hps.copy()
    SubtractingRightLeftMatching['name'] = 'TwoStimSubtractingRightLeftMatching'

    SummingMatching = generic_hps.copy()
    SummingMatching['name'] = 'TwoStimSummingMatching'

    SummingOppositeMatching = generic_hps.copy()
    SummingOppositeMatching['name'] = 'TwoStimSummingOppositeMatching'

    return [LeftIndicationMatching, RightIndicationMatching, \
        MinIndicationMatching, MaxIndicationMatching, \
        SubtractingLeftRightMatching, SubtractingRightLeftMatching, \
        SummingMatching, SummingOppositeMatching]


def twostim_tasks():

    generic_timing = {'dead_time'   : 300,
                      'fix_time'    : 200,
                      'sample_time'  : 300,
                      'delay_time'   : 1000,
                      'test_time'    : 300}
    generic_hps = {'n_motion_dirs': 8,
                   'n_sample': 2,
                   'n_RFs': 2,
                   'n_cues': 1, # No cues
                   'var_delay': False,
                   'distractor': False,
                   'n_output': 9, # N_DIR + 1 for fixation
                   'n_test': 0,
                   'mask_duration': 40,
                   'var_delay_max': 200,
                   'trial_length': sum(generic_timing.values()),
                   'timing': generic_timing}

    # All 2-stim tasks share the same set of meta-params
    Averaging = generic_hps.copy()
    Averaging['name'] = 'TwoStimAveraging'

    AveragingOpposite = generic_hps.copy()
    AveragingOpposite['name'] = 'TwoStimAveragingOpposite'

    LeftIndication = generic_hps.copy()
    LeftIndication['name'] = 'TwoStimLeftIndication'

    LeftIndicationOpposite = generic_hps.copy()
    LeftIndicationOpposite['name'] = 'TwoStimLeftIndicationOpposite'

    RightIndication = generic_hps.copy()
    RightIndication['name'] = 'TwoStimRightIndication'

    RightIndicationOpposite = generic_hps.copy()
    RightIndicationOpposite['name'] = 'TwoStimRightIndicationOpposite'

    MinIndication = generic_hps.copy()
    MinIndication['name'] = 'TwoStimMinIndication'

    MaxIndication = generic_hps.copy()
    MaxIndication['name'] = 'TwoStimMaxIndication'

    SubtractingLeftRight = generic_hps.copy()
    SubtractingLeftRight['name'] = 'TwoStimSubtractingLeftRight'

    SubtractingRightLeft = generic_hps.copy()
    SubtractingRightLeft['name'] = 'TwoStimSubtractingRightLeft'

    Summing = generic_hps.copy()
    Summing['name'] = 'TwoStimSumming'

    SummingOpposite = generic_hps.copy()
    SummingOpposite['name'] = 'TwoStimSummingOpposite'

    return [LeftIndication, RightIndication, \
        MinIndication, MaxIndication, SubtractingLeftRight, SubtractingRightLeft, Summing, SummingOpposite]

def revised_tasks():

    generic_timing = {'dead_time'   : 300,
                     'fix_time'     : 200,
                     'sample_time'  : 300,
                     'delay_time'   : 1000,
                     'test_time'    : 300,
                     'cue_time'     : 200}

    DMS = {}
    DMS['name'] = 'DMS'
    DMS['n_motion_dirs'] = 6
    DMS['n_cues'] = 0
    DMS['n_RFs'] = 1
    DMS['var_delay'] = False
    DMS['distractor'] = False
    DMS['n_output'] = 3
    DMS['var_delay_max'] = 200
    DMS['mask_duration'] = 40
    DMS['n_sample'] = 1
    DMS['n_test'] = 1
    DMS['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    DMS['timing'] = generic_timing

    AntiDMS = {}
    AntiDMS['name'] = 'AntiDMS'
    AntiDMS['n_motion_dirs'] = 6
    AntiDMS['rotation'] = 45
    AntiDMS['n_cues'] = 0
    AntiDMS['n_RFs'] = 1
    AntiDMS['var_delay'] = False
    AntiDMS['distractor'] = False
    AntiDMS['n_output'] = 3
    AntiDMS['var_delay_max'] = 200
    AntiDMS['mask_duration'] = 40
    AntiDMS['n_sample'] = 1
    AntiDMS['n_test'] = 1
    AntiDMS['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    AntiDMS['timing'] = generic_timing

    DMC = {}
    DMC['name'] = 'DMC'
    DMC['n_motion_dirs'] = 6
    DMC['n_cues'] = 0
    DMC['n_RFs'] = 1
    DMC['var_delay'] = False
    DMC['n_output'] = 3
    DMC['var_delay_max'] = 200
    DMC['mask_duration'] = 40
    DMC['n_sample'] = 1
    DMC['n_test'] = 1
    DMC['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    DMC['timing'] = generic_timing

    AntiDMC = {}
    AntiDMC['name'] = 'AntiDMC'
    AntiDMC['n_motion_dirs'] = 6
    AntiDMC['n_cues'] = 0
    AntiDMC['n_RFs'] = 1
    AntiDMC['var_delay'] = False
    AntiDMC['n_output'] = 3
    AntiDMC['var_delay_max'] = 200
    AntiDMC['mask_duration'] = 40
    AntiDMC['n_sample'] = 1
    AntiDMC['n_test'] = 1
    AntiDMC['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    AntiDMC['timing'] = generic_timing

    OIC = {}
    OIC['name'] = 'OIC'
    OIC['n_motion_dirs'] = 6
    OIC['n_cues'] = 0
    OIC['n_RFs'] = 1
    OIC['var_delay'] = False
    OIC['categorization'] = True
    OIC['n_output'] = 3
    OIC['var_delay_max'] = 200
    OIC['mask_duration'] = 40
    OIC['n_sample'] = 1
    OIC['n_test'] = 1
    OIC['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    OIC['timing'] = generic_timing

    AntiOIC = {}
    AntiOIC['name'] = 'AntiOIC'
    AntiOIC['n_motion_dirs'] = 6
    AntiOIC['n_cues'] = 0
    AntiOIC['n_RFs'] = 1
    AntiOIC['var_delay'] = False
    AntiOIC['categorization'] = True
    AntiOIC['n_output'] = 3
    AntiOIC['var_delay_max'] = 200
    AntiOIC['mask_duration'] = 40
    AntiOIC['n_sample'] = 1
    AntiOIC['n_test'] = 1
    AntiOIC['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    AntiOIC['timing'] = generic_timing

    return [DMS, AntiDMS, DMC, AntiDMC, OIC, AntiOIC]

def default_tasks(task_set):

    generic_timing = {'dead_time'   : 300,
                     'fix_time'     : 200,
                     'sample_time'  : 300,
                     'delay_time'   : 1000,
                     'test_time'    : 300,
                     'cue_time'     : 200}


    pro_ret_timing = copy.copy(generic_timing)
    pro_ret_timing['cue_time'] = 200

    abba_timing   = {'dead_time'    : 0,
                     'fix_time'     : 200,
                     'sample_time'  : 300,
                     'delay_time'   : 500,
                     'test_time'    : 300}

    # Example of usage: DMS, DMRS, DMC, DelayGo (all same extra params)
    DMS = {}
    DMS['name'] = 'DMS'
    DMS['n_motion_dirs'] = 6
    DMS['n_cues'] = 0
    DMS['n_RFs'] = 1
    DMS['var_delay'] = False
    DMS['distractor'] = False
    DMS['n_output'] = 3
    DMS['var_delay_max'] = 200
    DMS['mask_duration'] = 40
    DMS['n_sample'] = 1
    DMS['n_test'] = 1
    DMS['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    DMS['timing'] = generic_timing

    DMS_distractor = {}
    DMS_distractor['name'] = 'DMS'
    DMS_distractor['n_motion_dirs'] = 6
    DMS_distractor['n_cues'] = 0
    DMS_distractor['n_RFs'] = 1
    DMS_distractor['var_delay'] = False
    DMS_distractor['distractor'] = True
    DMS_distractor['n_output'] = 3
    DMS_distractor['var_delay_max'] = 200
    DMS_distractor['mask_duration'] = 40
    DMS_distractor['n_sample'] = 1
    DMS_distractor['n_test'] = 1
    DMS_distractor['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    DMS_distractor['timing'] = generic_timing

    DMRS45 = {}
    DMRS45['name'] = 'DMRS'
    DMRS45['rotation'] = 45
    DMRS45['n_motion_dirs'] = 6
    DMRS45['n_cues'] = 0
    DMRS45['n_RFs'] = 1
    DMRS45['var_delay'] = False
    DMRS45['distractor'] = False
    DMRS45['n_output'] = 3
    DMRS45['var_delay_max'] = 200
    DMRS45['mask_duration'] = 40
    DMRS45['n_sample'] = 1
    DMRS45['n_test'] = 1
    DMRS45['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    DMRS45['timing'] = generic_timing

    DMRS90 = {}
    DMRS90['name'] = 'DMRS'
    DMRS90['rotation'] = 90
    DMRS90['n_motion_dirs'] = 6
    DMRS90['n_cues'] = 0
    DMRS90['n_RFs'] = 1
    DMRS90['var_delay'] = False
    DMRS90['distractor'] = False
    DMRS90['n_output'] = 3
    DMRS90['var_delay_max'] = 200
    DMRS90['mask_duration'] = 40
    DMRS90['n_sample'] = 1
    DMRS90['n_test'] = 1
    DMRS90['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    DMRS90['timing'] = generic_timing

    DMRS180 = {}
    DMRS180['name'] = 'DMRS'
    DMRS180['rotation'] = 180
    DMRS180['n_motion_dirs'] = 6
    DMRS180['n_cues'] = 0
    DMRS180['n_RFs'] = 1
    DMRS180['var_delay'] = False
    DMRS180['distractor'] = False
    DMRS180['n_output'] = 3
    DMRS180['var_delay_max'] = 200
    DMRS180['mask_duration'] = 40
    DMRS180['n_sample'] = 1
    DMRS180['n_test'] = 1
    DMRS180['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    DMRS180['timing'] = generic_timing

    DMRS270 = {}
    DMRS270['name'] = 'DMRS'
    DMRS270['rotation'] = 270
    DMRS270['n_motion_dirs'] = 6
    DMRS270['n_cues'] = 0
    DMRS270['n_RFs'] = 1
    DMRS270['var_delay'] = False
    DMRS270['distractor'] = False
    DMRS270['n_output'] = 3
    DMRS270['var_delay_max'] = 200
    DMRS270['mask_duration'] = 40
    DMRS270['n_sample'] = 1
    DMRS270['n_test'] = 1
    DMRS270['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    DMRS270['timing'] = generic_timing

    DMC = {}
    DMC['name'] = 'DMC'
    DMC['n_motion_dirs'] = 6
    DMC['n_cues'] = 0
    DMC['n_RFs'] = 1
    DMC['var_delay'] = False
    DMC['n_output'] = 3
    DMC['var_delay_max'] = 200
    DMC['mask_duration'] = 40
    DMC['n_sample'] = 1
    DMC['n_test'] = 1
    DMC['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    DMC['timing'] = generic_timing


    DelayGo = {}
    DelayGo['name'] = 'DelayGo'
    DelayGo['n_motion_dirs'] = 6
    DelayGo['n_cues'] = 0
    DelayGo['n_RFs'] = 1
    DelayGo['var_delay'] = False
    DelayGo['categorization'] = True
    DelayGo['n_output'] = 3
    DelayGo['var_delay_max'] = 200
    DelayGo['mask_duration'] = 40
    DelayGo['n_sample'] = 1
    DelayGo['n_test'] = 1
    DelayGo['trial_length'] = sum(generic_timing.values()) - generic_timing['cue_time']
    DelayGo['timing'] = generic_timing

    ABBA = {}
    ABBA['name'] = 'ABBA'
    ABBA['n_motion_dirs'] = 6
    ABBA['n_cues'] = 0
    ABBA['n_RFs'] = 1
    ABBA['var_delay'] = False
    ABBA['n_output'] = 3
    ABBA['var_delay_max'] = 200
    ABBA['mask_duration'] = 40
    ABBA['n_sample'] = 1
    ABBA['n_test'] = 3
    ABBA['match_test_prob'] = 0.5
    ABBA['repeat_pct'] = 0.5
    ABBA['trial_length'] = 0
    for k, v in abba_timing.items():
        if 'delay' in k or 'test' in k:
            ABBA['trial_length'] += (ABBA['n_test'] * v)
        else:
            ABBA['trial_length'] += v
    ABBA['timing'] = abba_timing

    ABCA = {}
    ABCA['name'] = 'ABCA'
    ABCA['n_motion_dirs'] = 6
    ABCA['n_cues'] = 0
    ABCA['n_RFs'] = 1
    ABCA['var_delay'] = False
    ABCA['n_output'] = 3
    ABCA['var_delay_max'] = 200
    ABCA['mask_duration'] = 40
    ABCA['n_sample'] = 1
    ABCA['n_test'] = 3
    ABCA['match_test_prob'] = 0.5
    ABCA['repeat_pct'] = 0.0
    ABCA['trial_length'] = 0
    for k, v in abba_timing.items():
        if 'delay' in k or 'test' in k:
            ABCA['trial_length'] += (ABBA['n_test'] * v)
        else:
            ABCA['trial_length'] += v
    ABCA['timing'] = abba_timing

    ProRetroWM = {}
    ProRetroWM['name'] = 'ProRetroWM'
    ProRetroWM['n_motion_dirs'] = 6
    ProRetroWM['n_sample'] = 2
    ProRetroWM['n_cues'] = ProRetroWM['n_sample']
    ProRetroWM['n_RFs'] = ProRetroWM['n_sample']
    ProRetroWM['n_test'] = 1
    ProRetroWM['var_delay'] = False
    ProRetroWM['categorization'] = True
    ProRetroWM['n_output'] = 3
    ProRetroWM['var_delay_max'] = 200
    ProRetroWM['mask_duration'] = 40
    ProRetroWM['trial_length'] = 0
    for k, v in pro_ret_timing.items():
        if 'cue' not in k:
            ProRetroWM['trial_length'] += v
    ProRetroWM['timing'] = pro_ret_timing

    ProWM = {}
    ProWM['name'] = 'ProWM'
    ProWM['n_motion_dirs'] = 6
    ProWM['n_sample'] = 2
    ProWM['n_cues'] = ProWM['n_sample']
    ProWM['n_RFs'] = ProWM['n_sample']
    ProWM['n_test'] = 1
    ProWM['var_delay'] = False
    ProWM['categorization'] = True
    ProWM['n_output'] = 3
    ProWM['var_delay_max'] = 200
    ProWM['mask_duration'] = 40
    ProWM['trial_length'] = 0
    for k, v in pro_ret_timing.items():
        if 'cue' not in k:
            ProWM['trial_length'] += v
    ProWM['timing'] = pro_ret_timing

    RetroWM = {}
    RetroWM['name'] = 'RetroWM'
    RetroWM['n_motion_dirs'] = 6
    RetroWM['n_sample'] = 2
    RetroWM['n_cues'] = RetroWM['n_sample']
    RetroWM['n_RFs'] = RetroWM['n_sample']
    RetroWM['n_test'] = 1
    RetroWM['var_delay'] = False
    RetroWM['categorization'] = True
    RetroWM['n_output'] = 3
    RetroWM['var_delay_max'] = 200
    RetroWM['mask_duration'] = 40
    RetroWM['trial_length'] = 0
    for k, v in pro_ret_timing.items():
        if 'cue' not in k:
            RetroWM['trial_length'] += v
    RetroWM['timing'] = pro_ret_timing




    if task_set == "7tasks":
        task_list = [DMS, DMS_distractor, DMRS180, DMC, DelayGo, ProWM, RetroWM]
    elif task_set == "5tasks":
        task_list = [DMS, DMS_distractor, DMRS180, DMC, DelayGo]
    elif task_set == '2stim':
        task_list = twostim_tasks()
    elif task_set == '2stim_matching':
        task_list = twostim_matchingtasks()
    elif task_set == 'revised_tasks':
        task_list = revised_tasks()
    elif task_set == "challenge":
        task_list = [DMS, ABBA, ABCA]

    return task_list


if __name__ == "__main__":

    # Pull all task_list together
    task_list = default_tasks('2stim')
    tm    = TaskManager(task_list, batch_size=512)

    for rule_id in range(len(task_list)):
        print("\n\n")
        print(f"RULE {rule_id} ({task_list[rule_id]['name']})")
        b = tm.generate_batch(30, to_exclude=np.setdiff1d(np.arange(len(task_list)), rule_id))
        print("\n\n")
        