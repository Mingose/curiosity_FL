
#设置shell变量
cd ../
root_dir=$(pwd)
echo 'root_dir'$root_dir''

#生成数据集，并且切分为多个任务
#####################################################################
export n_components=1
export n_tasks=7
export alpha=0.5
# CUDA_VISIBLE_DEVICES=0 
# cd $root_dir/data/cifar100/
# echo "generate data"
echo 'train n_components='${n_components}' n_tasks='$n_tasks'  alpha='$alpha' imbalance_ratio='$imbalance_ratio''
# python generate_data.py --n_components $n_components --n_tasks $n_tasks --alpha $alpha --s_frac 1.0  --tr_frac 0.8   --seed 12345
# echo "generate data done"

#开始训练不平和程度为10倍的cifar100
###################################################################
export imbalance_ratio=0.1
cd $root_dir

export curi=1.0
while [ $(echo "$curi <5.0" | bc) -eq 1 ] #增加curi的条件检查
do
echo "run AvgCuri $curi"
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 1 --logs_dir logs/M11D14__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_99_JUSTLR2__curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler cosine_annealing --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.99  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_99_JUSTLR__cosine_annealing_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
curi=$(echo "$curi + 0.1" | bc)
done
####################################
export imbalance_ratio=0.02
export curi=1.0
while [ $(echo "$curi <5.0" | bc) -eq 1 ] #增加curi的条件检查
do
echo "run AvgCuri $curi"
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 1 --logs_dir logs/M11D14__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_099_JUSTLR2__curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler cosine_annealing --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.99  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_99_JUSTLR__cosine_annealing_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
curi=$(echo "$curi + 0.1" | bc)
done
#############################################
export imbalance_ratio=0.01
export curi=1.0
while [ $(echo "$curi <5.0" | bc) -eq 1 ] #增加curi的条件检查
do
echo "run AvgCuri $curi"
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 1  --logs_dir logs/M11D14__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_099_JUSTLR2__curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler cosine_annealing --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.99  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_99_JUSTLR__cosine_annealing_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
curi=$(echo "$curi + 0.1" | bc)
done


while [ $(echo "$curi <0.0" | bc) -eq 1 ] #增加curi的条件检查
do
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.70  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_70_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.71  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_71_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.72  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_72_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.73  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_73_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.74  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_74_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.75  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_75_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.76  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_76_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.77  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_77_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.78  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_78_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.79  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_79_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.80  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_80_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.81  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_81_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.82  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_82_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.83  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_83_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.84  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_84_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.85  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_85_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.86  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_86_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.87  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_87_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.88  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_88_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.89  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_89_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.90  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_90_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.91  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_91_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.92  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_92_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.93  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_93_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.94  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_94_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.95  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_95_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.96  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_96_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True

python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.97  --logs_dir logs/M11D12__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_97_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True


# curi=$(echo "$curi + 0.2" | bc)
# done