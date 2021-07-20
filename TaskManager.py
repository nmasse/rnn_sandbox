import numpy as np 
import copy
import tasks.DMRS
import tasks.DMC
import tasks.DelayGo
import tensorflow as tf
import matplotlib.pyplot as plt

class TaskManager:

    def __init__(self, 
                 task_list, 
                 batch_size,
                 n_motion_tuned = 32, 
                 n_fix_tuned = 1, 
                 tuning_height = 2, 
                 kappa = 2, 
                 dt = 20,
                 input_mean = 0,
                 input_noise = 0.1,
                 catch_trial_pct = 0., 
                 test_cost_mult = 1.,
                 fix_break_penalty = -1.0,
                 correct_choice_reward = 1.0,
                 wrong_choice_penalty = -0.01,
                 tf2 = False):
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

        # Filter for DMS among task_list specified
        for t in task_list:
            if t['name'] == 'DMS':
                t['name'] = 'DMRS'
                t['rotation'] = 0

        # Use number of task_list to determine number of rule-tuned inputs,
        # cue-tuned inputs
        self.n_rule_tuned = len(task_list)
        self.n_cue_tuned  = max([t['n_cues'] for t in task_list])
        
        # Determine number of RFs required, num motion directions; duplicate 
        # motion tuning for each receptive field
        self.n_RFs = max([t['n_RFs'] for t in task_list])
        self.n_motion_dirs = max([t['n_motion_dirs'] for t in task_list])
        self.n_motion_tuned *= self.n_RFs

        self.n_input = self.n_motion_tuned + \
                       self.n_rule_tuned   + \
                       self.n_cue_tuned    + \
                       self.n_fix_tuned
        self.n_output     = np.sum(np.unique([t['n_output'] for t in task_list]))
        self.trial_length = max([sum(t['timing'].values()) // self.dt for t in task_list])

        # Build tuning, shape
        self.tuning = self.create_tuning_functions()
        self.shape  = self.get_shape()

        # Build and bind task objects
        self.task_list, self.task_names = [], []
        for i, t in enumerate(task_list):
            self.task_names.append(t['name'])
            task = eval(f"tasks.{t['name']}.{t['name']}")

            # Create the task-specific dictionary of miscellaneous params
            misc = copy.copy(t)
            misc['input_mean']            = self.input_mean
            misc['input_noise']           = self.input_noise
            misc['catch_trial_pct']       = self.catch_trial_pct
            misc['test_cost_multiplier']  = self.test_cost_mult
            misc['fix_break_penalty']     = self.fix_break_penalty
            misc['correct_choice_reward'] = self.correct_choice_reward
            misc['wrong_choice_penalty']  = self.wrong_choice_penalty
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

            #self.dataset = self.dataset.cache()
            self.dataset = self.dataset.batch(self.batch_size)
            self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def visualize_task(self, rule_id):
        # Visualize neural inputs and desired outputs of specified task
        trial_info = self.generate_batch(1, rule=rule_id)
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 10))
        ax[0].imshow(trial_info['neural_input'].squeeze().T, aspect='equal')
        ax[1].imshow(trial_info['desired_output'].squeeze().T, aspect='equal')
        plt.show()


    def generate_batch_tf2(self):
        while True:
            yield self.generate_batch(1)


    def generate_batch(self, batch_size, rule=None, **kwargs):
        # Generate a batch of trials; if rule is specified, 
        # only generate trials of that rule, otherwise
        # generate at random from all task_list interleaved 
        # w/in same batch
        if rule is not None:
            trial_info = self.task_list[rule].generate_trials(batch_size, **kwargs)
        else:
            # Get empty trial info dictionary, then write its elements,
            # one at a time
            trial_info = self.generate_empty_trial_info(batch_size)
            for i in range(batch_size):
                rule = np.random.choice(self.n_rule_tuned)
                trial_info = self.write_trial(trial_info, 
                    self.task_list[rule].generate_trials(1, **kwargs), i, batch_size)

        # Extract just the elements that are necessary for the tf dataset
        if self.tf2:
            keys = ['neural_input', 'desired_output', 'train_mask', 
                'reward_matrix', 'sample', 'rule']
            return tuple([trial_info[k].squeeze() for k in keys])
            
        return trial_info

    def generate_empty_trial_info(self, batch_size):
        # Generate an empty trial info dictionary to add other trials to, one-by-one
        trial_info = {'desired_output'  :  np.zeros((batch_size, self.trial_length, self.n_output), dtype=np.float32),
                      'train_mask'      :  np.ones((batch_size, self.trial_length), dtype=np.float32),
                      'rule'            :  np.zeros((batch_size), dtype=np.int8),
                      'neural_input'    :  np.random.normal(self.input_mean, self.input_noise, 
                                                size=(batch_size, self.trial_length, self.n_input)),
                      'sample'          :  np.zeros((batch_size), dtype=np.int8),
                      'reward_matrix'   :  np.zeros((batch_size, self.trial_length, self.n_output), dtype=np.float32),
                      'timing'          : [],
                      'task_specific'   : []}

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
        shape['trial_length']   = self.trial_length
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
        fix_tuning    = np.zeros((self.n_input, 1))
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
            fix_tuning[self.n_motion_tuned + n, 0] = self.tuning_height

        ###
        # Generate rule tuning
        for n in range(self.n_rule_tuned):
            rule_tuning[self.n_motion_tuned+self.n_fix_tuned+n,n] = self.tuning_height

        ###
        # Generate cue tuning
        for n in range(self.n_cue_tuned):
            start = self.n_motion_tuned + self.n_fix_tuned + self.n_rule_tuned
            cue_tuning[start + n, n] = self.tuning_height

        # Package together into a dictionary and return
        tuning = {'motion': motion_tuning,
                  'fix'   : fix_tuning, 
                  'rule'  : rule_tuning,
                  'cue'   : cue_tuning}
        return tuning

def default_tasks():

    generic_timing = {'dead_time'    : 0,
                     'fix_time'     : 500,
                     'sample_time'  : 500,
                     'delay_time'   : 1000,
                     'test_time'    : 500}

    # Example of usage: DMS, DMRS, DMC, delay_go (all same extra params)
    DMS = {}
    DMS['name'] = 'DMS'
    DMS['n_motion_dirs'] = 8
    DMS['n_cues'] = 0
    DMS['n_RFs'] = 1
    DMS['var_delay'] = False
    DMS['n_output'] = 3
    DMS['var_delay_max'] = 200
    DMS['mask_duration'] = 50
    DMS['timing'] = generic_timing

    DMRS45 = {}
    DMRS45['name'] = 'DMRS'
    DMRS45['rotation'] = 45
    DMRS45['n_motion_dirs'] = 8
    DMRS45['n_cues'] = 0
    DMRS45['n_RFs'] = 1
    DMRS45['var_delay'] = False
    DMRS45['n_output'] = 3
    DMRS45['var_delay_max'] = 200
    DMRS45['mask_duration'] = 50
    DMRS45['timing'] = generic_timing

    DMRS90 = {}
    DMRS90['name'] = 'DMRS'
    DMRS90['rotation'] = 90
    DMRS90['n_motion_dirs'] = 8
    DMRS90['n_cues'] = 0
    DMRS90['n_RFs'] = 1
    DMRS90['var_delay'] = False
    DMRS90['n_output'] = 3
    DMRS90['var_delay_max'] = 200
    DMRS90['mask_duration'] = 50
    DMRS90['timing'] = generic_timing

    DMRS180 = {}
    DMRS180['name'] = 'DMRS'
    DMRS180['rotation'] = 180
    DMRS180['n_motion_dirs'] = 8
    DMRS180['n_cues'] = 0
    DMRS180['n_RFs'] = 1
    DMRS180['var_delay'] = False
    DMRS180['n_output'] = 3
    DMRS180['var_delay_max'] = 200
    DMRS180['mask_duration'] = 50
    DMRS180['timing'] = generic_timing

    DMRS270 = {}
    DMRS270['name'] = 'DMRS'
    DMRS270['rotation'] = 270
    DMRS270['n_motion_dirs'] = 8
    DMRS270['n_cues'] = 0
    DMRS270['n_RFs'] = 1
    DMRS270['var_delay'] = False
    DMRS270['n_output'] = 3
    DMRS270['var_delay_max'] = 200
    DMRS270['mask_duration'] = 50
    DMRS270['timing'] = generic_timing

    DMC = {}
    DMC['name'] = 'DMC'
    DMC['n_motion_dirs'] = 8
    DMC['n_cues'] = 0
    DMC['n_RFs'] = 1
    DMC['var_delay'] = False
    DMC['n_output'] = 3
    DMC['var_delay_max'] = 200
    DMC['mask_duration'] = 50
    DMC['timing'] = generic_timing

    delay_go = {}
    delay_go['name'] = 'DelayGo'
    delay_go['n_motion_dirs'] = 8
    delay_go['n_cues'] = 1
    delay_go['n_RFs'] = 1
    delay_go['var_delay'] = False
    delay_go['n_output'] = 8
    delay_go['var_delay_max'] = 200
    delay_go['mask_duration'] = 50
    delay_go['timing'] = generic_timing

    task_list = [DMS, DMRS45, DMRS90, DMRS180, DMRS270, DMC, delay_go]

    return task_list


if __name__ == "__main__":

    # Pull all task_list together
    task_list = default_tasks()
    tm    = TaskManager(task_list, batch_size=512)
    for rule_id in range(len(task_list)):
        tm.visualize_task(rule_id)
