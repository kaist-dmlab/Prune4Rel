# CIFAR10
for data in CIFAR100
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
          python3 ../main_label_noise.py --gpu 3 --model 'PreActResNet18' --noise-type $noise_type \
          -se 30 --epochs 300 --fraction $fraction --selection $method --save-log True --pre-warmuped False \
          --dataset $data --n-class 100 --lambda-u 1 \
          --metric cossim --uncertainty LeastConfidence \
          --tau 0.95 --eta 1 --balance True
        done
      done
    done
  done
done