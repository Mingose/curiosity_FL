
#设置shell变量
cd ../
root_dir=$(pwd)
echo 'root_dir'$root_dir''
export n_components=1
export n_tasks=10
cd $root_dir/data/imageNet_LT/
echo "generate data"
#传入参数--n_components=n_components
CUDA_VISIBLE_DEVICES=0 python  generate_data.py --n_components $n_components --n_tasks $n_tasks --alpha 0.5 --s_frac 1.0   --tr_frac 0.9  --seed 1234 
echo "generate data done"
# while [ $n_tasks -le 80 ] #这tasks设置10-110,每次加10
# do
cd $root_dir
echo 'train n_components='${n_components}' n_tasks='$n_tasks''
# echo "imageNet_LT Fedavg Mu=0.9"
# CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Fedavg_mu09 --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Fedavg_mu09   
echo "imageNet_LT Fedavg Mu=0.94"
CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.94 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Fedavg_mu092 --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Fedavg_mu092   
echo "imageNet_LT Fedavg Mu=0.96"
CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.96 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Fedavg_mu096 --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Fedavg_mu096  
echo "imageNet_LT Fedavg Mu=0.98"
CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.98 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Fedavg_mu098 --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Fedavg_mu098 
echo "imageNet_LT Fedavg Mu=1"
CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 1 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Fedavg_mu1 --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Fedavg_mu1

# echo "Run Clustered FL"
# CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT clustered --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.003 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Clustered --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_Clustered 
# echo "imageNet_LT FedProx"
# CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT FedProx  --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --momentum 0.9 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_FedProx --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_FedProx 
echo "imageNet_LT FedEM"
CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT FedEM --n_learners 3 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_leaner3 --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_FedEM_Leaner3   
# echo "imageNet_LT CasuleEM sgd"
# CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT FedEM --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_casuleEM --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_CasuleEM   
# echo "imageNet_LT Casule sgd"
# CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler WarmupMultiStepLR --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --momentum 0.9 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_casule --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_casule   

export curi=0.36
while [ $(echo "$curi < 0.4" | bc) -eq 1 ] #增加curi的条件检查
do
    echo "run AvgCuri $curi mu = 0.9~1"
    # python run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.92  --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU92_curi$curi --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU92_curiAvg_$curi --curi $curi
    # python run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.94  --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU94_curi$curi --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU94_curiAvg_$curi --curi $curi
    # python run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.96  --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU96_curi$curi --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU96_curiAvg_$curi --curi $curi
    python run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.98  --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU98_curi$curi --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU98_curiAvg_$curi --curi $curi
    python run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.99  --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU99_curi$curi --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU99_curiAvg_$curi --curi $curi
    python run_experiment.py imageNet_LT FedAvg --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 1  --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU1_curi$curi --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_MU1_curiAvg_$curi --curi $curi
    curi=$(echo "$curi + 0.01" | bc)
done

    # echo "run curiEM 0.30"
    # CUDA_VISIBLE_DEVICES=0 python  run_experiment.py imageNet_LT FedEM --n_learners 1 --decentralized --n_rounds 100 --bz 64 --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer lm_sgd --seed 1234 --verbose 1 --momentum 0.9 --save_dir output/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_curiEM0.30 --logs_dir logs/M10_D10_FixMu_imageNetLT_Decentrolized_just_traindata/components${n_components}_Alpha0.5_tasks${n_tasks}_curiEM0.30 --curi 0.30 
    # n_tasks=$((n_tasks+10))
# done
