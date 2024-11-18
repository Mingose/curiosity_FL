
#设置shell变量
# cd ../
# root_dir=$(pwd)
# echo 'root_dir'$root_dir''

# #生成数据集，并且切分为多个任务
# #####################################################################
# export n_components=1
# export n_tasks=7
# export alpha=0.5
# # CUDA_VISIBLE_DEVICES=0 
# # cd $root_dir/data/cifar100/
# # echo "generate data"
# echo 'train n_components='${n_components}' n_tasks='$n_tasks'  alpha='$alpha' imbalance_ratio='$imbalance_ratio''
# # python generate_data.py --n_components $n_components --n_tasks $n_tasks --alpha $alpha --s_frac 1.0  --tr_frac 0.8   --seed 12345
# # echo "generate data done"

# #开始训练不平和程度为10倍的cifar100
# ###################################################################
# export imbalance_ratio=0.1
# cd $root_dir

# export curi=1.0
# while [ $(echo "$curi <5.0" | bc) -eq 1 ] #增加curi的条件检查
# do
# echo "run AvgCuri $curi"
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.9 --logs_dir logs/M11D15__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_99_JUSTLR2__curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler cosine_annealing --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.99  --logs_dir logs_test2/M11D16__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_99_JUSTLR__cosine_annealing_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# curi=$(echo "$curi + 0.1" | bc)
# done
# ####################################
# export imbalance_ratio=0.02
# export curi=1.0
# while [ $(echo "$curi <5.0" | bc) -eq 1 ] #增加curi的条件检查
# do
# echo "run AvgCuri $curi"
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.9 --logs_dir logs/M11D15__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_099_JUSTLR2__curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler cosine_annealing --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.99  --logs_dir logs_test2/M11D16__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_99_JUSTLR__cosine_annealing_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# curi=$(echo "$curi + 0.1" | bc)
# done
# #############################################
# export imbalance_ratio=0.01
# export curi=1.0
# while [ $(echo "$curi <5.0" | bc) -eq 1 ] #增加curi的条件检查
# do
# echo "run AvgCuri $curi"
# python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs/M11D15__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_099_JUSTLR2__curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# # python run_experiment.py cifar100 FedAvg --n_learners 1 --n_rounds 100 --bz 1024 --lr 0.07 --lr_scheduler cosine_annealing --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.99  --logs_dir logs_test2/M11D16__cifar100_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR007_mu_99_JUSTLR__cosine_annealing_curiAvg_$curi --curi $curi --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# curi=$(echo "$curi + 0.1" | bc)
# done

####################################################################################################################################
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
# python generate_data.py --n_components $n_components --n_tasks $n_tasks --alpha $alpha --s_frac 1.0  --tr_frac 0.8  --seed 12345
echo "generate data done"
export imbalance_ratio=0.1
cd $root_dir
# echo "run cifar10 Avg_warmup lr0.04"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.04 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR004_tasks${n_tasks}_Avg_warmup --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.04 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_Fedavg_LR004_Focalloss   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True --use_focalloss
# echo "Run cifar10 Fedavg momentu 09"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_Fedavg_LR001_Mu09 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_FedProx"
# python  run_experiment.py cifar10 FedProx  --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_LR001_Fedavg_FedProx --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Clustered FL"
# python  run_experiment.py cifar10 clustered --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_LR001_Clustered --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_RSloss lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_Fedavg_LR001_RSloss   --use_RSloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True --use_RSloss
# echo "Run cifar10 FedEM"
# python  run_experiment.py cifar10 FedEM --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_FedEM_LR001   --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True

# export curi=0.25
# while [ $(echo "$curi <2" | bc) -eq 1 ] #增加curi的条件检查
# do
# echo "run AvgCuri $curi bz256"
# echo "run AvgCuri AVG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
# python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.04 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR004_MUL01_mu_99_JUSTLR__cosine_annealing_curiAvg_$curi --curi $curi --curi_lr 0.1 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# curi=$(echo "$curi + 0.04" | bc)
# done


####################################
export imbalance_ratio=0.02
cd $root_dir
# echo "run cifar10 Avg_warmup lr0.04"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.04 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR004_tasks${n_tasks}_Avg_warmup --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_Focalloss lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.04 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_Fedavg_LR004_Focalloss   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True --use_focalloss
# echo "Run cifar10 Fedavg momentu 09"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_Fedavg_LR001_Mu09 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_FedProx"
# python  run_experiment.py cifar10 FedProx  --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_LR001_Fedavg_FedProx --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Clustered FL"
# python  run_experiment.py cifar10 clustered --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_LR001_Clustered --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
# echo "Run cifar10 Fedavg_RSloss lr0.03"
# python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_Fedavg_LR001_RSloss   --use_RSloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True --use_RSloss
# echo "Run cifar10 FedEM"
# python  run_experiment.py cifar10 FedEM --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_FedEM_LR001   --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True

export curi=0.12
while [ $(echo "$curi <2" | bc) -eq 1 ] #增加curi的条件检查
do
echo "run AvgCuri $curi bz256"
echo "run AvgCuri AVG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.04 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR004_MUL01_mu_99_JUSTLR__cosine_annealing_curiAvg_$curi --curi $curi --curi_lr 0.1 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
curi=$(echo "$curi + 0.04" | bc)
done
#############################################
export imbalance_ratio=0.01
cd $root_dir
echo "run cifar10 Avg_warmup lr0.04"
python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.04 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components${n_components}_ResNet32_LR004_tasks${n_tasks}_Avg_warmup --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar10 Fedavg_Focalloss lr0.03"
python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.04 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_Fedavg_LR004_Focalloss   --use_focalloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True --use_focalloss
echo "Run cifar10 Fedavg momentu 09"
python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_Fedavg_LR001_Mu09 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar10 Fedavg_FedProx"
python  run_experiment.py cifar10 FedProx  --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_LR001_Fedavg_FedProx --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar10 Clustered FL"
python  run_experiment.py cifar10 clustered --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9   --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_LR001_Clustered --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
echo "Run cifar10 Fedavg_RSloss lr0.03"
python  run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_Fedavg_LR001_RSloss   --use_RSloss --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True --use_RSloss
echo "Run cifar10 FedEM"
python  run_experiment.py cifar10 FedEM --n_learners 1 --n_rounds 100 --bz 256 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_FedEM_LR001   --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True

export curi=0.01
while [ $(echo "$curi <2" | bc) -eq 1 ] #增加curi的条件检查
do
echo "run AvgCuri $curi bz256"
echo "run AvgCuri AVG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 100 --bz 256 --lr 0.04 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.9  --logs_dir logs_test2/M11D17__cifar10_IB${imbalance_ratio}_Alpha${alpha}/components1_tasks${n_tasks}_ResNet32_LR004_MUL01_mu_99_JUSTLR__cosine_annealing_curiAvg_$curi --curi $curi --curi_lr 0.1 --imbalance_ratio $imbalance_ratio --use_imbalance_ratio True
curi=$(echo "$curi + 0.04" | bc)
done