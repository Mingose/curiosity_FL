from re import X
import time

from sklearn.inspection import PartialDependenceDisplay

from models import *
from datasets import *
from learners.learner import *
from learners.learners_ensemble import *
from client import *
from aggregator import *



from .ResNet32Feature import *
from .optim import *
from .metrics import *
from .constants import *
from .decentralized import *

from .dataloader import *

from torch.utils.data import DataLoader

from tqdm import tqdm

from utils_old.args import *

from .loss import FocalLoss
from .loss import ReweightedSoftmaxLoss


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment

    :param experiment_name: name of the experiment
    :return: str

    """
    data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def get_learner(
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        momentum,
        curi,
        input_dim=None,
        output_dim=None,
        lr_factor=None,
        warm_epoch=None
):
    """
    构造对应于给定种子的实验的学习器

    :param name：要使用的实验名称；可能是
                 {`合成`，`cifar10`，`emnist`，`莎士比亚`}
    :param device: 使用的设备；可能的`cpu`和`cuda`
    :param optimizer_name:优化器名称--作为参数传递给 utils.optim.get_optimizer
    :param scheduler_name: 作为参数传递给 utils.optim.get_lr_scheduler
    :param initial_lr: 学习率的初始值
    :param mu: 近端权重，仅在 `optimizer_name=="prox_sgd"` 时使用
    :param input_dim: 输入维度，仅用于合成数据集
    :param output_dim: output_dimension;仅用于合成数据集
    :param n_rounds: 训练轮数，仅在 `scheduler_name == multi_step` 时使用，默认为 None；
    ：参数种子：
    :return: 学习者

    """
    torch.manual_seed(seed)

    args = parse_args()

    if name == "synthetic":
        if output_dim == 2:
            criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
            metric = binary_accuracy
            model = LinearLayer(input_dim, 1).to(device)
            is_binary_classification = True
        else:
            if args.use_focalloss:
                print('use Focal Loss')
                criterion = FocalLoss(alpha=args.Focal_alpha, gamma=args.Focal_gamma)
            else:
                criterion = nn.CrossEntropyLoss(reduction="none").to(device)
            metric = accuracy
            model = LinearLayer(input_dim, output_dim).to(device)
            is_binary_classification = False
            
    elif name == "cifar10":
        if args.use_focalloss:
            print('use Focal Loss')
            criterion = FocalLoss(alpha=args.Focal_alpha, gamma=args.Focal_gamma)
        elif args.use_RSloss:
            print('Use Reweighted Softmax Loss')
            criterion = ReweightedSoftmaxLoss(weights=[1.0 for _ in range(10)]) #10个类不知道权重，只能设置全为1
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        # model = get_mobilenet(n_classes=10).to(device)
        # model = get_resnet18(n_classes=10).to(device)
        
        if scheduler_name == "WarmupMultiStepLR":
            print("BBN_ResNet32_cifar10 lr_factor:{} WarmupMultiStepLR_warm_up:{}".format(lr_factor,warm_epoch))    
            model = create_model()
            # model = get_resnet34(n_classes=10).to(device)
        else:
            # print('cifar10_use_mobi')
            # model = get_mobilenet(n_classes=10).to(device)
            print('cifar10_use_resnet34')
            model = get_resnet34(n_classes=10).to(device)
            # print('cifar10_use_resnet18')
            # model = get_resnet18(n_classes=10).to(device)
        is_binary_classification = False

    elif name == "cifar100":
        # print(f'scheduler_name:{scheduler_name}')
        if args.use_focalloss:
            print('use Focal Loss')
            criterion = FocalLoss(alpha=args.Focal_alpha, gamma=args.Focal_gamma)
        elif args.use_RSloss:
            print('Use Reweighted Softmax Loss')
            criterion = ReweightedSoftmaxLoss(weights=[1.0 for _ in range(100)]) 
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        # model = create_model()
        # model = get_mobilenet(n_classes=100).to(device)
        if scheduler_name == "WarmupMultiStepLR":
            print('cifar100_use_resnet34')
            print("cifar100 lr_factor:{} warm_up:{}".format(lr_factor,warm_epoch))
            model = create_model()
        else:
            # print('cifar100_use_mobi')
            # model = get_mobilenet(n_classes=100).to(device)
            print('cifar100_use_resnet34')
            model = get_resnet34(n_classes=100).to(device)
            # print('cifar100_use_resnet18')
            # model = get_resnet18(n_classes=100).to(device)
        is_binary_classification = False
    elif name == "emnist" or name == "femnist":
        if args.use_focalloss:
            print('use Focal Loss')
            criterion = FocalLoss(alpha=args.Focal_alpha, gamma=args.Focal_gamma)
        elif args.use_RSloss:
            print('Use Reweighted Softmax Loss')
            criterion = ReweightedSoftmaxLoss(weights=[1.0 for _ in range(62)]) 
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        model = FemnistCNN(num_classes=62).to(device)
        is_binary_classification = False
    elif name == "shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8

        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
        metric = accuracy
        model =\
            NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"]
            ).to(device)
        is_binary_classification = False


    elif name =="imageNet_LT":
        if args.use_focalloss:
            print('use Focal Loss')
            criterion = FocalLoss(alpha=args.Focal_alpha, gamma=args.Focal_gamma)
        elif args.use_RSloss:
            print('Use Reweighted Softmax Loss')
            criterion = ReweightedSoftmaxLoss(weights=[1.0 for _ in range(1000)]) 
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        if scheduler_name == "WarmupMultiStepLR":
            print("imageNet_LT lr_factor:{} WarmupMultiStepLR_warm_up:{}".format(lr_factor,warm_epoch))    
            model = create_model_imageNet()
            # model = get_resnet101(n_classes=1000).to(device)
            # print('imageNet_LT_use_Res101')
            # model = get_resnet34(n_classes=1000).to(device)
            # print('imageNet_LT_use_Res34')
        else:
            # model = get_mobilenet(n_classes=1000).to(device)
            # print('imageNet_LT_use_mobi')
            # model = get_resnet18(n_classes=1000).to(device)
            # print('imageNet_LT_use_Res18')
            # model = get_resnet101(n_classes=1000).to(device)
            # print('imageNet_LT_use_Res101')
            # model = get_resnet34(n_classes=1000).to(device)
            model = get_resnet50(n_classes=1000).to(device)
            print('imageNet_LT_use_Res50')
            # print('imageNet_LT_use_Res34')
            # model = get_resnet34(n_classes=1000).to(device)
        
        is_binary_classification = False
    elif name == "meter":
        if args.use_focalloss:
            print('use Focal Loss')
            criterion = FocalLoss(alpha=args.Focal_alpha, gamma=args.Focal_gamma)
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        if scheduler_name == "WarmupMultiStepLR":
            print("imageNet_LT lr_factor:{} WarmupMultiStepLR_warm_up:{}".format(lr_factor,warm_epoch))    
            model = create_model()
        else:
            # model = get_mobilenet(n_classes=1000).to(device)
            model = get_resnet18(n_classes=10).to(device)
            print('imageNet_LT_use_mobi')
            # model = get_resnet18(n_classes=10).to(device)

        is_binary_classification = False



    else:
        raise NotImplementedError

    optimizer =\
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=initial_lr,
            mu=mu,
            momentum=momentum ,#动量
            curi=curi #好奇
        )
    lr_scheduler =\
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds,
            lr_factor=lr_factor,
            warm_epoch=warm_epoch
        )

    if name == "shakespeare":
        return LanguageModelingLearner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    else:
        # return Learner(
        #     model=model,
        #     criterion=criterion,
        #     metric=metric,
        #     device=device,
        #     optimizer=optimizer,
        #     lr_scheduler=lr_scheduler,
        #     is_binary_classification=is_binary_classification
        # )
        learner = Learner(
            model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification
        )
    # 打印模型形状
    # print(f"Model architecture: {model}")

    # # 打印优化器类型
    # print(f"Optimizer type: {optimizer.__class__.__name__}")

    # 打印优化器的动量衰减参数
    if isinstance(optimizer, optim.SGD):
        print(f"Optimizer momentum: {optimizer.param_groups[0]['momentum']}")
    elif "momentum" in optimizer.defaults:
        print(f"Optimizer momentum: {optimizer.defaults['momentum']}")
    else:
        print("This optimizer does not have a momentum parameter.")

    return learner


def get_learners_ensemble(
        n_learners,
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        momentum,
        curi,
        input_dim=None,
        output_dim=None,
        lr_factor=None,
        warm_epoch=None
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param n_learners: number of learners in the ensemble
    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: LearnersEnsemble

    """

    learners = [
        get_learner(
            name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed + learner_id,
            mu=mu,
            momentum=momentum,
            curi=curi,
            lr_factor=lr_factor,
            warm_epoch=warm_epoch
        ) for learner_id in range(n_learners)
    ]

    learners_weights = torch.ones(n_learners) / n_learners
    if name == "shakespeare":
        return LanguageModelingLearnersEnsemble(learners=learners, learners_weights=learners_weights)
    else:
        return LearnersEnsemble(learners=learners, learners_weights=learners_weights)


def get_loaders(type_, root_path, batch_size, is_validation,imbalance_ratio=0.0,use_imbalance_ratio=False):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last

    :param type_: type of the dataset;
    :param root_path: path to the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])

    """
    if type_ == "cifar10":
        inputs, targets = get_cifar10()
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
    elif type_ == "emnist":
        inputs, targets = get_emnist()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = [], [], []
    print("type_:{}".format(type_))

    if  type_ =="imageNet_LT" or type_ == "meter":
        print("load txt format  data")
        for task_id, task_dir in enumerate(tqdm(os.listdir(root_path))):
            task_data_path = os.path.join(root_path, task_dir)

            train_iterator = \
                load_data(
                data_root=task_data_path,
                dataset=type_,
                batch_size=batch_size,
                phase="train",
                shuffle=False,

                )
            val_iterator = \
                load_data(
                    data_root=task_data_path,
                    dataset=type_,
                    batch_size=batch_size,
                    phase="val",
                    shuffle=False
                )

            test_iterator = \
                load_data(
                    data_root=task_data_path,
                    dataset=type_,
                    batch_size=batch_size,
                    phase="test",
                    shuffle=False
                )
            
            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)
    else:

        for task_id, task_dir in enumerate(tqdm(os.listdir(root_path))):
            task_data_path = os.path.join(root_path, task_dir)

            train_iterator = \
                get_loader(
                    type_=type_,
                    path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                    batch_size=batch_size,
                    inputs=inputs,
                    targets=targets,
                    train=True,
                    imbalance_ratio=imbalance_ratio,
                 use_imbalance_ratio=use_imbalance_ratio
            )

            val_iterator = \
                get_loader(
                    type_=type_,
                    path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                    batch_size=batch_size,
                    inputs=inputs,
                    targets=targets,
                    train=False,
                    imbalance_ratio=imbalance_ratio,
                    use_imbalance_ratio=False
            )

            if is_validation:
                test_set = "val"
            else:
                test_set = "test"

            test_iterator = \
                get_loader(
                    type_=type_,
                    path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                    batch_size=batch_size,
                    inputs=inputs,
                    targets=targets,
                    train=False,
                    imbalance_ratio=imbalance_ratio,
                    use_imbalance_ratio=False
            )

            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)
    # print('Len(train_iterators)=',len(train_iterators))
    # print('Len(val_iterators)=',len(val_iterators))
    # print('Len(test_iterators)=',len(test_iterators))


    return train_iterators, val_iterators, test_iterators


def get_loader(type_, path, batch_size, train,imbalance_ratio=0.0, inputs=None, targets=None,use_imbalance_ratio=False):
    """
    constructs a torch.utils.DataLoader object from the given path

    :param type_: type of the dataset; possible are `tabular`, `images` and `text`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return: torch.utils.DataLoader

    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "cifar10":
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets,imbalance_ratio=imbalance_ratio,use_imbalance_ratio=use_imbalance_ratio)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets,imbalance_ratio=imbalance_ratio,use_imbalance_ratio=use_imbalance_ratio)
    elif type_ == "emnist":
        dataset = SubEMNIST(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = ((type_ == "cifar100") or (type_ == "cifar10")) and (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)


def get_client(
        client_type,
        learners_ensemble,
        q,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally,
):
    """

    :param client_type:
    :param learners_ensemble:
    :param q: fairness hyper-parameter, ony used for FFL client
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param tune_locally

    :return:

    """
    if client_type == "mixture":
        return MixtureClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
    elif client_type == "AFL":
        return AgnosticFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
    elif client_type == "FFL":
        return FFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            q=q
        )
    else:
        return Client(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )


def get_aggregator(
        aggregator_type,
        clients,
        global_learners_ensemble,
        lr,
        lr_lambda,
        mu,
        communication_probability,
        q,
        sampling_rate,
        log_freq,
        global_train_logger,
        global_test_logger,
        test_clients,
        verbose,
        seed=None
):
    """
    `personalized` corresponds to pFedMe

    :param aggregator_type:
    :param clients:
    :param global_learners_ensemble:
    :param lr: oly used with FLL aggregator
    :param lr_lambda: only used with Agnostic aggregator
    :param mu: penalization term, only used with L2SGD
    :param communication_probability: communication probability, only used with L2SGD
    :param q: fairness hyper-parameter, ony used for FFL client
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    if aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "centralized":
        print("aggregator_type == centralized")
        return CentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "personalized":
        return PersonalizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "clustered":
        return ClusteredAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "L2SGD":
        return LoopLessLocalSGDAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            communication_probability=communication_probability,
            penalty_parameter=mu,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "AFL":
        return AgnosticAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr_lambda=lr_lambda,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "FFL":
        return FFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr=lr,
            q=q,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "decentralized":
        n_clients = len(clients)

        mixing_matrix = get_mixing_matrix(n=n_clients, p=0.7, seed=seed)

        return DecentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            mixing_matrix=mixing_matrix,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    else:
        raise NotImplementedError(
            "{aggregator_type} is not a possible aggregator type."
            " Available are: `no_communication`, `centralized`,"
            " `personalized`, `clustered`, `fednova`, `AFL`,"
            " `FFL` and `decentralized`."
        )
