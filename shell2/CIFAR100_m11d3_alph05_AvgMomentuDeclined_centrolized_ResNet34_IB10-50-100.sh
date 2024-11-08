#设置shell变量
cd ../
root_dir=$(pwd)
echo 'root_dir'$root_dir''

#生成数据集，并且切分为多个任务
#####################################################################
export n_components=1
export n_tasks=7
export alpha=0.5
CUDA_VISIBLE_DEVICES=0 
cd $root_dir/data/cifar100/
echo "generate data"
echo 'train n_components='${n_components}' n_tasks='$n_tasks'  alpha='$alpha' imbalance_ratio='$imbalance_ratio''
python generate_data.py --n_components $n_components --n_tasks $n_tasks --alpha $alpha --s_frac 1.0  --tr_frac 0.8   --seed 12345
echo "generate data done"

#开始训练不平和程度为10倍的CIFAR10
###################################################################
export imbalance_ratio=0.1
cd $root_dir
echo "run cifar100 Avg_warmup"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmuplr0.01 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg_FedProx"
python  run_experiment.py cifar100 FedProx  --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_FedProx --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Clustered FL"
# python  run_experiment.py cifar100 clustered --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Clustered --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "run cifar100 Avg_warmup lr0.03"
# python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR003_tasks${n_tasks}_Avg_warmup_lr0.03 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 Fedavg_Focalloss lr0.03"
# python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.03   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 Fedavg_RSloss lr0.03"
# python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_RSloss_lr0.03   --use_RSloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 FedEM"
# python  run_experiment.py cifar100 FedEM --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_FedEM   --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True

echo "Run cifar100 Fedavg momentu 081"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.81   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu081 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 082"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.82   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu082 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 083"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.83   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu083 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 084"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.84   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu084 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 085"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.85   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu085 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 086"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.86  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu086 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 087"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.87  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu087 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 088"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.88   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu88 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 089"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.89   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu89 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 Fedavg momentu 090"
# python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu09 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 091"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.91   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu091 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 092"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.92   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu092 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 093"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.93   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu093 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 094"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.94   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu094 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 095"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.95   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu095 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 096"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.96   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu096 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 097"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.97   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu097 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 098"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.98   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu098 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 099"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.99   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu099 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True



export curi=0.20
while [ $(echo "$curi <0.38 " | bc) -eq 1 ] #增加curi的条件检查
do
    echo "run AvgCuri $curi"
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.96  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_96_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.94  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_94_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.92  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_92_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.91  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_91_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.90  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_90_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.96  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR003_mu_96_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.94  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR003_mu_94_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.92  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR003_mu_92_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.91  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR003_mu_91_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.90  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR003_mu_90_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    curi=$(echo "$curi + 0.02" | bc)
done

#开始训练不平和程度为50倍的CIFAR10
###################################################################
export imbalance_ratio=0.02
export imbalance_ratio=0.02
export imbalance_ratio=0.02
export imbalance_ratio=0.02
export imbalance_ratio=0.02
cd $root_dir
# echo "run cifar100 Avg_warmup"
# python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmuplr0.01 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 Fedavg_FedProx"
# python  run_experiment.py cifar100 FedProx  --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_FedProx --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 Clustered FL"
# python  run_experiment.py cifar100 clustered --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Clustered --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "run cifar100 Avg_warmup lr0.03"
# python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR003_tasks${n_tasks}_Avg_warmup_lr0.03 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 Fedavg_Focalloss lr0.03"
# python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.03   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 Fedavg_RSloss lr0.03"
# python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_RSloss_lr0.03   --use_RSloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 FedEM"
# python  run_experiment.py cifar100 FedEM --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_FedEM   --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True

echo "Run cifar100 Fedavg momentu 081"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.81   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu081 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 082"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.82   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu082 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 083"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.83   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu083 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 084"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.84   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu084 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 085"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.85   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu085 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 086"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.86  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu086 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 087"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.87  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu087 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 088"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.88   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu88 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 089"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.89   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu89 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 Fedavg momentu 090"
# python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu09 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 091"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.91   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu091 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 092"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.92   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu092 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 093"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.93   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu093 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 094"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.94   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu094 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 095"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.95   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu095 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 096"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.96   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu096 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 097"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.97   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu097 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 098"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.98   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu098 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 099"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.99   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu099 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True

export curi=0.20
while [ $(echo "$curi <0.36 " | bc) -eq 1 ] #增加curi的条件检查
do
    echo "run AvgCuri $curi"
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.96  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_96_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.94  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_94_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.92  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_92_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.91  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_91_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.90  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_90_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.96  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR003_mu_96_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.94  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR003_mu_94_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.92  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR003_mu_92_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.91  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR003_mu_91_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.90  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR003_mu_90_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    curi=$(echo "$curi + 0.02" | bc)
done


#开始训练不平和程度为100倍的CIFAR10
###################################################################
export imbalance_ratio=0.01
export imbalance_ratio=0.01
export imbalance_ratio=0.01
export imbalance_ratio=0.01
cd $root_dir
echo "run cifar100 Avg_warmup"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR001_tasks${n_tasks}_Avg_warmuplr0.01 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg_FedProx"
python  run_experiment.py cifar100 FedProx  --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_FedProx --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Clustered FL"
python  run_experiment.py cifar100 clustered --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Clustered --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "run cifar100 Avg_warmup lr0.03"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.03 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR003_tasks${n_tasks}_Avg_warmup_lr0.03 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg_Focalloss lr0.03"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_Focalloss_lr0.03   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg_RSloss lr0.03"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_LR001_RSloss_lr0.03   --use_RSloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 FedEM"
python  run_experiment.py cifar100 FedEM --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_FedEM   --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True


echo "Run cifar100 Fedavg momentu 081"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.81   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu081 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 082"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.82   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu082 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 083"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.83   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu083 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 084"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.84   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu084 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 085"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.85   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu085 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 086"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.86  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu086 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 087"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.87  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu087 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 088"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.88   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu88 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 089"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.89   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu89 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar100 Fedavg momentu 090"
# python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu09 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 091"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.91   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu091 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 092"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.92   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu092 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 093"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.93   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu093 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 094"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.94   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu094 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 095"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.95   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu095 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 096"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.96   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu096 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 097"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.97   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu097 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 098"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.98   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu098 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar100 Fedavg momentu 099"
python  run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.99   --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_tasks${n_tasks}_Fedavg_Mu099 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
export curi=0.20
while [ $(echo "$curi <0.36 " | bc) -eq 1 ] #增加curi的条件检查
do
    echo "run AvgCuri $curi"
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.92  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_92_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.91  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_91_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.90  --logs_dir logs/M11D3__CIFAR100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR001_mu_90_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
    curi=$(echo "$curi + 0.02" | bc)
done