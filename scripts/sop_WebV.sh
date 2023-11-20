# CIFAR10
for data in WebVision
do
  for seed in 0 1 2
  do
    for fraction in 0.2 0.4 0.6 0.8
    do
      # Uniform Uncertainty kCenterGreedy Forgetting GraNd Glister Moderate / SmallLoss / Prune4Rel
      for method in Prune4Rel
      do
        for noise_type in 'real'
        do
          python3 ../main_label_noise.py --gpu 0 --model 'Inception_ResNetv2' --robust-learner 'SOP' -rc $seed -rb 0 \
          --noise-type 'clean' -se 10 --epochs 100 --fraction $fraction --selection $method --save-log True --pre-warmuped False \
          --dataset $data --data_path '/path/to/data/' --n-class 50 --lr-u 0.1 --lr-v 1 \
          --scheduler "MultiStepLR" --batch-size 32 --n-train 65944 --metric cossim --uncertainty LeastConfidence \
          --tau 0.95 --eta 1 --balance True
        done
      done
    done
  done
done