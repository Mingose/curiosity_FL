#设置shell变量
cd ../
root_dir=$(pwd)
echo 'root_dir'$root_dir''

#生成数据集，并且切分为多个任务
#####################################################################
export n_components=1
export n_tasks=10
export alpha=0.5
CUDA_VISIBLE_DEVICES=0 
cd $root_dir/data/cifar10/
echo "generate data"
echo 'train n_components='${n_components}' n_tasks='$n_tasks'  alpha='$alpha' imbalance_ratio='$imbalance_ratio''
python generate_data.py --n_components $n_components --n_tasks $n_tasks --alpha $alpha --s_frac 1.0  --tr_frac 0.8  --seed 12345
echo "generate data done"

#开始训练不平和程度为10倍的CIFAR10
###################################################################
export imbalance_ratio=0.1
cd $root_dir
# echo "run cifar10 Avg_warmup"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmup --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_FedProx"
# python  run_experiment.py cifar10 FedProx  --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_FedProx --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Clustered FL"
# python  run_experiment.py cifar10 clustered --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Clustered --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "run cifar10 Avg_warmup lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmup_lr0.03 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "run cifar10 Avg_warmup lr0.005"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.005 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmup_lr0.005 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.01"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.01   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.03   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.005"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.005   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_RSloss lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_RSloss_lr0.03   --use_RSloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 FedEM"
# python  run_experiment.py cifar10 FedEM --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_FedEM   --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar10 Fedavg momentu 09"
python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu09 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 08"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.8   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu08 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 07"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.7   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu07 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 06"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.6   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu06 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 05"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.5   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu05 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 04"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.4   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu04 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.3   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu03 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 02"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.2   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu02 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 01"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.1   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu01 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 0"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.0   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu00 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True

export curi=0.14
while [ $(echo "$curi <0.4 " | bc) -eq 1 ] #增加curi的条件检查
do
    echo "run AvgCuri $curi"
    python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.96  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_96__curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.92  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_92_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.94  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_94_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.98  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_98_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.90  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_90_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.88  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_88_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer   lm_sgd --seed 1234 --verbose 1 --momentum 0.86  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_86_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    curi=$(echo "$curi + 0.02" | bc)
done

#开始训练不平和程度为50倍的CIFAR10
###################################################################
# export imbalance_ratio=0.02
# export imbalance_ratio=0.02
# export imbalance_ratio=0.02
# export imbalance_ratio=0.02
# export imbalance_ratio=0.02
# cd $root_dir
# echo "run cifar10 Avg_warmup"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmup --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_FedProx"
# python  run_experiment.py cifar10 FedProx  --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_FedProx --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Clustered FL"
# python  run_experiment.py cifar10 clustered --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Clustered --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "run cifar10 Avg_warmup lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmup_lr0.03 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "run cifar10 Avg_warmup lr0.005"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.005 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmup_lr0.005 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.01"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.01   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.03   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.005"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.005   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_RSloss lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_RSloss_lr0.03   --use_RSloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 FedEM"
# python  run_experiment.py cifar10 FedEM --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_FedEM   --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 09"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu09 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 08"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.8   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu08 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 07"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.7   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu07 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 06"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.6   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu06 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 05"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.5   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu05 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 04"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.4   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu04 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.3   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu03 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 02"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.2   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu02 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 01"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.1   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu01 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 0"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.0   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu00 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True

# export curi=0.26
# while [ $(echo "$curi < 0.39" | bc) -eq 1 ] #增加curi的条件检查
# do
#     echo "run AvgCuri $curi"
#     python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.96  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
#     curi=$(echo "$curi + 0.01" | bc)
# done


#开始训练不平和程度为100倍的CIFAR10
###################################################################
# export imbalance_ratio=0.01
# export imbalance_ratio=0.01
# export imbalance_ratio=0.01
# export imbalance_ratio=0.01
# cd $root_dir
# echo "run cifar10 Avg_warmup"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmup --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_FedProx"
# python  run_experiment.py cifar10 FedProx  --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_FedProx --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Clustered FL"
# python  run_experiment.py cifar10 clustered --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Clustered --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "run cifar10 Avg_warmup lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmup_lr0.03 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "run cifar10 Avg_warmup lr0.005"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.005 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmup_lr0.005 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.01"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.01   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.03   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.005"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.005   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_RSloss lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_RSloss_lr0.03   --use_RSloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 FedEM"
# python  run_experiment.py cifar10 FedEM --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_FedEM   --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 09"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu09 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 08"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.8   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu08 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 07"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.7   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu07 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 06"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.6   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu06 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 05"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.5   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu05 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 04"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.4   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu04 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.3   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu03 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 02"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.2   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu02 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 01"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.1   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu01 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg momentu 0"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.0   --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu00 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True

# export curi=0.26
# while [ $(echo "$curi < 0.39" | bc) -eq 1 ] #增加curi的条件检查
# do
#     echo "run AvgCuri $curi"
#     python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.96  --logs_dir logs/M10D31_2__CIFAR10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
#     curi=$(echo "$curi + 0.01" | bc)
# done