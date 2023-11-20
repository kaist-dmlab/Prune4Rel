# CIFAR10
for data in Clothing1M
do
  for seed in 0 1 2
  do
    for fraction in 0.01 0.05 0.1 0.2 #0.01 0.05 0.1 0.2
    do
      # Uniform Uncertainty kCenterGreedy Forgetting GraNd Glister Moderate / SmallLoss / Prune4Rel
      for method in Prune4Rel
      do
        python3 ../main_label_noise.py --gpu 0 1 2 3 --model 'ResNet50' --dataset $data \
        -se 1 --epochs 10 --fraction $fraction --selection $method --save-log True \
        --data_path '/data/sachoi/Robustdata/clothing1m' --resolution 224 --batch-size 32 \
        --lr 0.002 --lambda-u 0.1 --alpha 0.5 -wd 1e-3 --pre-trained True --multi-gpu True \
        --scheduler "MultiStepLR" --noise-type 'clean' --n-train 1000000 --pre-warmuped True\
        --metric cossim --uncertainty LeastConfidence \
        --tau 0.8 --eta 1 --balance True
      done
    done
  done
done