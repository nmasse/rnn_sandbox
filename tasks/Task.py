import numpy as np

class Task(object):

    def __init__(self, task_name, rule_id, var_delay, dt, tuning, timing, shape, misc):

        # Bind generic task-identifying information
        self.task_name  = task_name
        self.rule_id    = rule_id 
        self.var_delay  = var_delay
        self.dt         = dt

        # Bind tuning to motion/fixation/rule/cue for generating inputs
        self.motion_tuning = tuning['motion'] 
        self.fix_tuning    = tuning['fix'] 
        self.rule_tuning   = tuning['rule'] 
        self.cue_tuning    = tuning['cue'] 

        # Store basic timing structure in # timesteps 
        for key, val in timing.items():
            setattr(self, key, timing[key] // dt)

        # Store basic network shape information
        self.n_input        = shape['n_input']
        self.n_output       = shape['n_output']
        self.trial_length   = shape['trial_length']
        self.n_motion_tuned = shape['n_motion_tuned']
        self.n_fix_tuned    = shape['n_fix_tuned']
        self.n_rule_tuned   = shape['n_rule_tuned']
        self.n_cue_tuned    = shape['n_cue_tuned']

        # Bind noise parameters for input generation, as well as catch trial pct and other misc. 
        # parameters
        self.input_mean           = misc['input_mean']
        self.input_noise          = misc['input_noise']
        self.catch_trial_pct      = misc['catch_trial_pct']
        self.test_cost_multiplier = misc['test_cost_multiplier']
        self.n_motion_dirs        = misc['n_motion_dirs']
        self.n_RFs                = misc['n_RFs']
        self.var_delay_max        = misc['var_delay_max']
        self.mask_duration        = misc['mask_duration']

        # Bind RL info
        self.fix_break_penalty     = misc['fix_break_penalty']
        self.correct_choice_reward = misc['correct_choice_reward']
        self.wrong_choice_penalty  = misc['wrong_choice_penalty']

    def _get_trial_info(self, batch_size):

        trial_info = {'desired_output'  :  np.zeros((batch_size, self.trial_length, self.n_output), dtype=np.float32),
                      'train_mask'      :  np.ones((batch_size, self.trial_length), dtype=np.float32),
                      'sample'          :  np.zeros((batch_size), dtype=np.int8),
                      'test'            :  np.zeros((batch_size), dtype=np.int8),
                      'rule'            :  np.zeros((batch_size), dtype=np.int8),
                      'match'           :  np.zeros((batch_size), dtype=np.int8),
                      'catch'           :  np.zeros((batch_size), dtype=np.int8),
                      'neural_input'    :  np.random.normal(self.input_mean, self.input_noise, 
                                                size=(batch_size, self.trial_length, self.n_input)),
                      'reward_matrix'   :  np.zeros((batch_size, self.trial_length, self.n_output), dtype=np.float32),
                      'timing'          : []}
        return trial_info