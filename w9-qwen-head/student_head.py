"""本人的核心：句对特征，以及把训练特征和标签交给分类头。"""
import numpy as np


def make_pair_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """把两句的固定向量变为对称句对特征。

    输入：left、right 均为 float32 的 [N,D] 数组。
          第 i 行属于同一个句对的左句与右句，N>0、D>0，无 NaN/Inf。
    输出：float32 的 [N,2*D] 数组，每行仍对应原来的那个句对。
          前 D 列为逐维绝对差，后 D 列为逐维乘积。
    不变量：交换左右句结果不变；不得修改输入、改变行顺序或使用标签。
    边界：输入检查由外围完成，本人先处理合法输入。

    查阅接口：np.abs、逐元素乘法、np.concatenate(..., axis=1)。
    axis=1 表示横向接列，保留 N 行。请根据上述职责自行组合。
    不需要计算注意力、余弦、梯度，也不需要重新编码句子。
    """
    abs_array=np.abs(left-right)
    multiply_array=np.multiply(left,right)
    return np.concatenate((abs_array,multiply_array),axis=1)


def fit_head(classifier, train_features: np.ndarray, train_labels: np.ndarray):
    """用 train 数据拟合已配置好的逻辑回归，并返回同一个 classifier。

    train_features: [N,2048]，每行一个句对（真实 Qwen 向量 D=1024）。
    train_labels: [N]，0/1 标签，行顺序与特征一致，两类都存在。
    classifier: 外围配置好的 sklearn LogisticRegression 对象。
    输出：已经 fit 的同一个 classifier。

    查阅接口：classifier.fit(X, y)。
    这里只用 train 数据；不要把诊断 dev 的标签交给 fit。
    本周只训练分类头，Qwen 主干和已缓存的句向量不会被更新。
    """
    classifier.fit(train_features,train_labels)
    return classifier
