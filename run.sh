#!/bin/bash


for method in lstm dnn #transformer
do
for loss in mse ll 
do
for speed in speed speed_limit speed_avg_week_time speed_avg
do
for type in 0 1
do
    echo "Do ->> $method $loss $speed $type"
    python ./main.py --model $method --datadir al --epochs 40 --lossfunc $loss --optimizer adam --batchsize 128 --summed $type --outputdir output --name v3 --speedprofile $speed
done
done
done
done


