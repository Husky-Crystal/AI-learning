import numpy as np

'''
简化决策树：
X：(m样本, n特征)
y：(m, )：二分类问题，类别用0/1表示
特征用 0~n-1 表示，每个特征只有0/1两种取值
'''

# 计算样本的信息熵
def compute_entropy(y):

    if len(y) == 0:
        return 0.
    
    p1 = np.mean(y == 1)
    if p1 == 0 or p1 == 1:
        return 0.
    
    return -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))


# 根据特征取值划分位于某一节点的样本集，特征取值为1的划分到左子树，0的划分到右子树
def split_dataset(X, node_indices, feature):
    left_indices = np.intersect1d(node_indices, np.argwhere(X[:,feature] == 1).ravel()).tolist()
    right_indices = np.intersect1d(node_indices, np.argwhere(X[:,feature] == 0).ravel()).tolist()        
    return left_indices, right_indices


# 计算某一节点上，按照某一特征划分的信息增益
def compute_information_gain(X, y, node_indices, feature):
    
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    h_node = compute_entropy(y_node)
    h_left = compute_entropy(y_left)
    h_right = compute_entropy(y_right)
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    return h_node - (w_left * h_left + w_right * h_right)


# 计算某一节点上信息增益最大的特征，用于划分样本
def get_best_split(X, y, node_indices):   
    
    num_features = X.shape[1]
    best_feature = -1
    max_information_gain = 0.
    
    for feature in range(num_features):
        information_gain = compute_information_gain(X, y, node_indices, feature)
        if information_gain > max_information_gain:
            best_feature = feature
            max_information_gain = information_gain 
   
    return best_feature


class dectree:

    left = None           # 左子树：该特征取值为1的样本
    right = None          # 右子树：该特征取值为0的样本
    feature_index = None  # 内部节点：根据哪个特征划分子树
    leaf_result = None    # 叶子节点：分类结果
    is_leaf = None        # 是否叶子节点

    def __init__(self, l, r, fi, lr, il):
        self.left = l
        self.right = r
        self.feature_index = fi
        self.leaf_result = lr
        self.is_leaf = il

    def show_info():
        pass


# 求众数
def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]


# 递归函数
def f(X, y, max_depth, node_indices, cur_depth):

    # 没有样本了
    if len(node_indices) == 0:
        return None

    # 当前层数超过阈值
    if cur_depth > max_depth:
        return dectree(None, None, None, mode(y[node_indices]), True)
    
    # 正常处理
    best_feature = get_best_split(X, y, node_indices)
    left_idx, right_idx = split_dataset(X, node_indices, best_feature)
    left_tree = f(X, y, max_depth, left_idx, cur_depth + 1)
    right_tree = f(X, y, max_depth, right_idx, cur_depth + 1)

    if left_tree == None or right_tree == None:
        return dectree(None, None, None, mode(y[node_indices]), True)
    else:
        return dectree(left_tree, right_tree, best_feature, None, False)


# 构建一棵简化版的决策树
def my_build_tree(X, y, max_depth):
    if X == None or y == None or len(X) == 0 or len(y) == 0 or len(X) != len(y) or max_depth <= 0:
        return None
    else:
        return f(X, y, max_depth, np.arange(len(y)), 0)

