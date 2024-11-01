import cvxpy as cp
import networkx as nx
import numpy as np
# import CVXOPT


def get_communication_graph(n, p, seed):
    return nx.generators.random_graphs.binomial_graph(n=n, p=p, seed=seed)


def compute_mixing_matrix(adjacency_matrix):
    """
    computes the mixing matrix associated to a graph defined by its `adjacency_matrix` using
    FMMC (Fast Mixin Markov Chain), see https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf

    :param adjacency_matrix: np.array()
    :return: optimal mixing matrix as np.array()
    """
    network_mask = 1 - adjacency_matrix
    N = adjacency_matrix.shape[0]

    s = cp.Variable()
    W = cp.Variable((N, N))
    objective = cp.Minimize(s)

    constraints = [
        W == W.T,   #确保混合矩阵 W 是对称的。
        W @ np.ones((N, 1)) == np.ones((N, 1)), #确保混合矩阵的每一行之和为1。
        cp.multiply(W, network_mask) == np.zeros((N, N)),   #强制网络中没有边的地方混合矩阵的对应元素为0
        -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
        W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N), #通过这两个约束，将 W 限制在一定范围内，以确保最小化目标函数 s
        np.zeros((N, N)) <= W   #确保混合矩阵中的元素都是非负的
    ]

    prob = cp.Problem(objective, constraints)   #创建一个 CVXPY 问题实例，包括目标函数和约束条件。
    prob.solve()    #求解问题，找到使得目标函数最小化的混合矩阵
    # prob.solve(solver=cp.CVXOPT)  # 尝试不同的求解器，例如 CVXOPT
    

    #调试mixing_matrix即W=None的问题：
    print("优化状态:", prob.status)
    mixing_matrix = W.value #获取求解得到的混合矩阵的值
    # print('mixing_matrix:',mixing_matrix)

    # print('adjacency_matrix:',adjacency_matrix)
    mixing_matrix *= adjacency_matrix

    mixing_matrix = np.multiply(mixing_matrix, mixing_matrix >= 0)  #将混合矩阵的负值置为0，以确保所有元素都是非负的

    # Force symmetry (added for numerical stability)
    for i in range(N):
        if np.abs(np.sum(mixing_matrix[i, i:])) >= 1e-20:
            mixing_matrix[i, i:] *= (1 - np.sum(mixing_matrix[i, :i])) / np.sum(mixing_matrix[i, i:])
            mixing_matrix[i:, i] = mixing_matrix[i, i:]

    return mixing_matrix


def get_mixing_matrix(n, p, seed):
    print('num of client{}通过概率为{}的厄多斯-雷尼图相连:'.format(n,p))

    graph = get_communication_graph(n, p, seed)
    adjacency_matrix = nx.adjacency_matrix(graph, weight=None).todense()

    return compute_mixing_matrix(adjacency_matrix)
