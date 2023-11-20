# CIFAR10
for data in ImageNet
do
  for seed in 0 1 2
  do
    for fraction in 0.05 0.1 0.2 0.4
    do
      # Uniform Uncertainty kCenterGreedy Forgetting GraNd Glister Moderate / SmallLoss / Prune4Rel
      for method in Prune4Rel
      do
        python3 main_label_noise.py --gpu 0 1 --model 'ResNet50' --resolution 224 \
        -se 10 --epochs 50 --batch-size 64 --fraction $fraction --selection $method --save-log True \
        --robust-learner 'SOP' -rc 0 -rb 0 --lr-u 0.1 --lr-v 1 --lr 0.02 -wd 1e-3 \
         --dataset $data --data_path '/path/to/data/' \
        --noise-type 'asym' --noise-rate 0.2 --multi-gpu True --test-batch-size 64 \
        --n-train 1281167 --pre-warmuped True --pre-warmed-filename '/path/to/warmup_model/' \
        --metric cossim --uncertainty LeastConfidence --tau 0.95 --balance True
      done
    done
  done
done