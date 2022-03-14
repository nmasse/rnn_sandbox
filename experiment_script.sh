# Experiment 1: Reliability (run 50 replicates of network for each good seed)
echo "Experiment 1:"
start_time=$SECONDS
python3 -m experiments.train --n_networks=50 --save_path='/media/graphnets/reservoir/experiment1/' \
    > logging/experiment1.txt 2>&1
elapsed=$(( SECONDS - start_time ))
day=$(($elapsed/3600/24))
d=$(date -ud "@$elapsed" +' days %H hr %M min %S sec')
echo $day $d

# Experiment 2: Activity generation (train 1 replicate per good seed, and save activity)
echo "Experiment 2:"
start_time=$SECONDS
python3 -m experiments.train --save_activities=True --save_path='/media/graphnets/reservoir/experiment2/' \
    > logging/experiment2.txt 2>&1
elapsed=$(( SECONDS - start_time ))
day=$(($elapsed/3600/24))
d=$(date -ud "@$elapsed" +' days %H hr %M min %S sec')
echo $day $d

# Experiment 3: Biological learning (train 2 networks per good seed)
echo "Experiment 3:"
start_time=$SECONDS
python3 -m experiments.train --n_networks=2 --training_alg='DNI' --n_training_iterations=5000 \
    --n_stim_batches=1000 --n_evaluation_iterations=100 --batch_size=64 --learning_rate=1e-4 \
    --save_path='/media/graphnets/reservoir/experiment3/' \
    > logging/experiment3.txt 2>&1
elapsed=$(( SECONDS - start_time ))
day=$(($elapsed/3600/24))
d=$(date -ud "@$elapsed" +' days %H hr %M min %S sec')
echo $day $d

# Experiment 4: Output quenching (make output not trainable, 1 networks per seed)
echo "Experiment 4:"
start_time=$SECONDS
python3 -m experiments.train --n_networks=2 --readout_trainable=False \
    --save_path='/media/graphnets/reservoir/experiment4' \
    --n_top_down_hidden=256 \
    --n_training_iterations=5000 \
    --n_stim_batches=1000 \
    > logging/experiment4.txt 2>&1
elapsed=$(( SECONDS - start_time ))
day=$(($elapsed/3600/24))
d=$(date -ud "@$elapsed" +' days %H hr %M min %S sec')
echo $day $d

# Experiment 5: Random initialization of top-down weights (10 networks per seed)
echo "Experiment 5:"
start_time=$SECONDS
python3 -m experiments.train --n_networks=10 --top_down_overlapping=False --top_down_trainable=False \
    --save_path='/media/graphnets/reservoir/experiment5/' \
    > logging/experiment5.txt 2>&1
elapsed=$(( SECONDS - start_time ))
day=$(($elapsed/3600/24))
d=$(date -ud "@$elapsed" +' days %H hr %M min %S sec')
echo $day $d

# Experiment 6: Fixed top-down weights (10 networks per seed)
echo "Experiment 6:"
start_time=$SECONDS
python3 -m experiments.train --n_networks=10 --top_down_trainable=False \
    --save_path='/media/graphnets/reservoir/experiment6' \
    > logging/experiment6.txt 2>&1
elapsed=$(( SECONDS - start_time ))
day=$(($elapsed/3600/24))
d=$(date -ud "@$elapsed" +' days %H hr %M min %S sec')
echo $day $d

# Experiment 7: Analysis (run anaylsis of saved activity)
echo "Experiment 7:"
start_time=$SECONDS
python3 -m analyses.activity_analyses > logging/experiment1.txt 2>&1
elapsed=$(( SECONDS - start_time ))
day=$(($elapsed/3600/24))
d=$(date -ud "@$elapsed" +' days %H hr %M min %S sec')
echo $day $d
