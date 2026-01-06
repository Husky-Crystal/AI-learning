'''
简化决策树：
X：(m样本, n特征)
y：(m, )：二分类问题，类别用0/1表示
特征用 0~n-1 表示，每个特征只有0/1两种取值

已实现：
0/1 特征, 二分类问题, ID3算法, 最大深度限制

未实现：
Gini系数, C4.5算法, CART算法
不重复使用特征, 连续特征
预剪枝, 后剪枝
随机森林, 集成算法, AdaBoost, XGBoost
'''

import numpy as np

# 计算样本的信息熵
def compute_entropy(y):

    if len(y) == 0:
        return 0.0
    
    p1 = np.mean(y == 1)
    if p1 == 0 or p1 == 1:
        return 0.0
    
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
    best_feature = None
    max_information_gain = 0.0
    
    for feature in range(num_features):
        information_gain = compute_information_gain(X, y, node_indices, feature)
        if information_gain > max_information_gain:
            best_feature = feature
            max_information_gain = information_gain 
   
    return best_feature


# 决策树节点类
class dectree:      

    def __init__(self, left, right, feature_index, leaf_result, is_leaf):
        self.left = left                     # 左子树：该特征取值为1的样本
        self.right = right                   # 右子树：该特征取值为0的样本
        self.feature_index = feature_index   # 内部节点：根据哪个特征划分子树
        self.leaf_result = leaf_result       # 叶子节点：分类结果，0/1类别
        self.is_leaf = is_leaf               # 是否叶子节点

    def show_info():
        pass


# 求众数
def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]


# 递归函数，node_indices是位于该节点的样本
def f(X, y, max_depth, node_indices, cur_depth):

    # 没有样本了，返回None
    if len(node_indices) == 0:
        return None

    # 当前层数超过阈值，返回叶节点，选择样本最多的类别作为该叶节点的标签
    if cur_depth >= max_depth:
        return dectree(None, None, None, mode(y[node_indices]), True)
    
    # 如果当前样本已经是同一类，返回叶节点，标签为该类
    if np.all(y[node_indices] == y[node_indices][0]):
        return dectree(None, None, None, mode(y[node_indices]), True)

    # 正常处理，先获取最优划分特征（信息增益最大）
    best_feature = get_best_split(X, y, node_indices)
    
    # 如果没有有效划分的特征，返回叶节点，选择样本最多的类别作为该叶节点的标签
    if best_feature is None:
        return dectree(None, None, None, mode(y[node_indices]), True)
    
    # 根据最优特征划分左右子树的样本
    left_idx, right_idx = split_dataset(X, node_indices, best_feature)
    
    # 如果左右子树其中一个为空，返回叶节点，选择样本最多的类别作为该叶节点的标签
    if len(left_idx) == 0 or len(right_idx) == 0:
        return dectree(None, None, None, mode(y[node_indices]), True)
    
    # 递归处理左右子树
    left_tree = f(X, y, max_depth, left_idx, cur_depth + 1)
    right_tree = f(X, y, max_depth, right_idx, cur_depth + 1)
   
    # 否则返回中间节点
    return dectree(left_tree, right_tree, best_feature, None, False)


# 构建一棵简化版的决策树
def train(X, y, max_depth) -> dectree:
    if X is None or y is None or len(X) == 0 or len(y) == 0 or len(X) != len(y) or max_depth <= 0:
        return None
    else:
        return f(X, y, max_depth, np.arange(len(y)), 0)


# 单个样本预测递归函数
def g(x, node: dectree):

    # 没有可用的特征
    if x is None or len(x) == 0:
        return None
    
    # 已到叶节点，返回预测结果
    if node.is_leaf:
        return node.leaf_result
    
    # 特征为1往左子树找；为0往右子树找
    if x[node.feature_index] == 1:
        return g(x, node.left)
    else:
        return g(x, node.right)


# 批量预测多个样本
def predict(X, tree: dectree):
    if tree is None:
        return None
    else:
        return np.array([g(x, tree) for x in X])


# 主函数
if __name__ == '__main__':

    X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
    y_train = np.array([1,1,0,0,1,0,0,1,1,0])
    max_depth = 2
    
    print('原始数据集：[特征] -> [类别]')
    for x, y in zip(X_train, y_train):
        print(f'{x} -> {y}')

    tree = train(X_train, y_train, max_depth)
    y_preds = predict(X_train, tree)
    print('')

    if y_preds is None:
        print('[Error] The decision is None!')
    else:
        print('预测结果：[特征] -> [类别], [预测], [T/F]')
        for x, y_true, y_pred in zip(X_train, y_train, y_preds):
            print(f'{x} -> {y_true}, {y_pred}, {y_true == y_pred}')

        print(f'预测准确率: {(np.mean(y_train == y_preds) * 100):.2f} %')
