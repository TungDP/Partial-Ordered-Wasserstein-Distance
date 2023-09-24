#! /bin/bash
# for metric in drop_dtw dtw softdtw pow dtw opw; do
#     for k in 1 3 5 7; do
#         for seed in 1 2 3 4 5; do
#             for distance in cityblock cosine ; do
#                 python -m src.experiments.weizmann.knn_eval --test_size 0.5 --outlier_ratio 0.3 --metric $metric --m 0.7 --reg 1 --distance $distance --k $k --seed $seed
#             done
#         done
#     done
# done
for outlier_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    #assign value m = 1 - outlier_ratio
    m=$(echo "1 - $outlier_ratio" | bc)
    # echo experiment with outlier_ratio $outlier_ratio and m $m
    echo "experiment with outlier_ratio $outlier_ratio and m $m"
    python -m src.experiments.weizmann.knn_eval --test_size 0.5 --outlier_ratio $outlier_ratio --metric opw --m $m --reg 1 --distance cosine --k 3 --seed 5
done
