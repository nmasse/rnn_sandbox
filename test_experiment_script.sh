# Experiment 1: Reliability (run 50 replicates of network for each good seed)
echo "Experiment 1:"
start_time=$SECONDS
python3 -m experiments.train --n_networks=1 --n_training_iterations=2 --n_evaluation_iterations=2 --batch_size=64 \
    --save_path='/media/graphnets/reservoir/experiment1/' > logging/experiment1.txt 2>&1
elapsed=$(( SECONDS - start_time ))
day=$(($elapsed/3600/24))
d=$(date -ud "@$elapsed" +' days %H hr %M min %S sec')
echo $day $d

# Experiment 2: Activity generation (train 1 replicate per good seed, and save activity)
echo "Experiment 2:"
python3 -m experiments.train --n_training_iterations=2 --n_evaluation_iterations=2 --batch_size=64 \
    --save_activities=True --save_path='/media/graphnets/reservoir/experiment2/' > logging/experiment2.txt 2>&1
echo "    finished."

# Experiment 3: Biological learning (train 10 networks per good seed)
echo "Experiment 3:"
python3 -m experiments.train --n_networks=1 --training_alg='DNI' --n_training_iterations=2 \
    --n_evaluation_iterations=2 --batch_size=64 --learning_rate=1e-4 \
    --save_path='/media/graphnets/reservoir/experiment3/' > logging/experiment3.txt 2>&1
echo "    finished."

# Experiment 4: Output quenching (make output not trainable, 10 networks per seed)
echo "Experiment 4:"
python3 -m experiments.train --n_networks=1 --readout_trainable=False --n_training_iterations=2 --n_evaluation_iterations=2 --batch_size=64 \
    --save_path='/media/graphnets/reservoir/experiment4' > logging/experiment4.txt 2>&1
echo "    finished."

# Experiment 5: Random initialization of top-down weights (10 networks per seed)
echo "Experiment 5:"
python3 -m experiments.train --n_networks=1 --top_down_overlapping=False --top_down_trainable=False --n_training_iterations=2 --n_evaluation_iterations=2 --batch_size=64 \
    --save_path='/media/graphnets/reservoir/experiment5/' > logging/experiment5.txt 2>&1
echo "    finished."

# Experiment 6: Fixed top-down weights (10 networks per seed)
echo "Experiment 6:"
python3 -m experiments.train --n_networks=1 --top_down_trainable=False --n_training_iterations=2 --n_evaluation_iterations=2 --batch_size=64 \
    --save_path='/media/graphnets/reservoir/experiment6' > logging/experiment6.txt 2>&1
echo "    finished."

# Experiment 7: Analysis (run anaylsis of saved activity)
echo "Experiment 7:"
python3 -m analyses.activity_analyses > logging/experiment7.txt 2>&1
echo "    finished."