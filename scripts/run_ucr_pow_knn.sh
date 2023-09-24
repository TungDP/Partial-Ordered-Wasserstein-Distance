#!/bin/bash

# for k in 1 3 5 7; do
#   for dataset in 'Chinatown' 'SmoothSubspace' 'ItalyPowerDemand' 'SonyAIBORobotSurface1' 'Coffee' 'SonyAIBORobotSurface2' 'Fungi' 'GunPoint' 'Plane' 'MoteStrain' 'FaceFour' 'OliveOil' 'DistalPhalanxTW' 'ToeSegmentation2' 'DistalPhalanxOutlineCorrect' 'Herring' 'BME' 'ECG200' 'BeetleFly' 'BirdChicken'; do
#     python -m src.experiments.ucr.knn_eval --outlier_ratio 0.2 --metric pow --m 0.8 --reg 100 --distance euclidean --k $k --dataset $dataset
#   done
# done

#!/bin/bash

# for k in 1 3 5 7; do
#   for dataset in 'Chinatown' 'SmoothSubspace' 'ItalyPowerDemand' 'SonyAIBORobotSurface1' 'Coffee' 'SonyAIBORobotSurface2' 'Fungi' 'GunPoint' 'Plane' 'MoteStrain' 'FaceFour'; do
#     python -m src.experiments.ucr.knn_eval --outlier_ratio 0.2 --metric pow --m 0.8 --reg 10 --distance euclidean --k $k --dataset $dataset
#   done
# done

#!/bin/bash


for metric in drop_dtw dtw softdtw pow dtw opw; do
  for dataset in 'BME' 'BeetleFly' 'BirdChicken' 'Chinatown' 'Coffee' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'ECG200' 'FaceFour' 'Fungi' 'GunPoint' 'Herring' 'ItalyPowerDemand' 'MoteStrain' 'OliveOil' 'Plane' 'SmoothSubspace' 'SonyAIBORobotSurface1' 'SonyAIBORobotSurface2' 'ToeSegmentation2'; do
    for seed in 1 2 3 4 5; do
        python -m src.experiments.ucr.knn_eval --outlier_ratio 0.2 --metric $metric --m 0.8 --reg 100 --distance euclidean --k 1 --dataset $dataset --seed $seed
    done
  done
done
