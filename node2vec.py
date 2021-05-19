from collections import defaultdict
import random
from pyspark.ml.feature import CountVectorizer, Word2Vec, Word2VecModel
import numpy as np
import math


class GraphWalk(object):
    def _one_walk(self):
        raise NotImplementedError

    def simulate_walks(self):
        raise NotImplementedError

    def pick_fisrt(self):
        """如何选择第一个样本
        1、根据所有item作为出发结点的频率来进行抽样，出发结点也即pair中的第一个元素
        2、根据所有item作为出发结点的频率来进行抽样，但是能确保每个样本都抽取到
        """
        if self.method == 'random':  # 根据概率来随机抽样，这样可能会使得部分样本抽不到
            prob_first = random.random()
            cur_item = ""
            accumulate_prob = 0.
            for item, prob in self.item_distributions.items():
                accumulate_prob += prob
                if (accumulate_prob >= prob_first):
                    break
            return item
        else:  # 先获取所有起始结点集合，然后遍历此集合中的结点作为出发位点
            if self.starters:
                return self.starters.pop()

    def simulate_walks(self):
        """采样
        """
        samples = []
        for i in range(self.sample_num):
            first_item = self.pick_fisrt()
            samples.append(self._one_walk(first_item))

        return samples


class DeepWalk(GraphWalk):
    def __init__(self, transfer_matrix, item_distributions, sample_length, sample_num, method='iter'):
        """ 随机游走，利用轮盘法来进行采样每个id
        Args:
            transfer_matrix: 嵌套字典，每个item的下一个item的概率分布，已归一化；
            item_distributions: 字典，原始所有pair的起始item的概率分布，已归一化；
            sample_length: int，每次采样长度；
            sample_num: int, 采样生成的sentence数量；
            method: str, 是否保证每个样本都被采到
        """
        self.transfer_matrix = transfer_matrix
        self.item_distributions = item_distributions
        self.sample_length = sample_length
        self.sample_num = sample_num

        self.method = method
        if self.method != 'random':
            # 生成序列的起始位点集合，确保每个结点都能被采样到
            self.starters = [i[0] for i in item_distributions.items()
                             for _ in range(int(math.ceil(sample_num*i[1])))]
            self.sample_num = len(self.starters)

    def _one_walk(self, first_item):
        """单次采样，使用轮盘法来选择每个item.
        """
        sample = [first_item]
        cur_item = first_item

        cnt = 1
        while (cnt <= self.sample_length):
            if (cur_item not in self.transfer_matrix) or (cur_item not in self.item_distributions):
                break
            next_distributions = self.transfer_matrix[cur_item]
            random_prob = random.random()
            accumulate_prob = 0.
            for item, prob in next_distributions.items():
                accumulate_prob += prob
                if (accumulate_prob >= random_prob):
                    cur_item = item
                    sample.append(cur_item)
                    break

            cnt += 1

        return sample


class Node2vec(GraphWalk):
    def __init__(self, transfer_matrix, item_distributions, sample_length, sample_num, p, q, method='iter'):
        """ node2vec
        Args:
            transfer_matrix: 嵌套字典，每个item的下一个item的概率分布，已归一化，
                    eg., {'n1':{'n2':0.1, 'n3':0.2, 'n4':0.7}, 'n3':{...}, ...}；
            item_distributions: 字典，原始所有pair的起始item的概率分布，已归一化；
            sample_length: int，每次采样长度；
            sample_num: int, 采样生成的sentence数量；
            p: float, 往回走的系数;
            q: float, DFS的系数, BFS的系数为1;
            method: str, 是否保证每个样本都被采到；
        """

        self.transfer_matrix = transfer_matrix
        self.item_distributions = item_distributions
        self.sample_length = sample_length
        self.sample_num = sample_num
        self.p = p
        self.q = q

        self.alias_edges = {}

        self.method = method
        if self.method != 'random':
            # 生成序列的起始位点集合，确保每个结点都能被采样到
            self.starters = [i[0] for i in item_distributions.items()
                             for _ in range(int(math.ceil(sample_num*i[1])))]
            self.sample_num = len(self.starters)

    def create_alias_table(self, area_ratio):
        """创建alias table，从而进行拒绝采样. 
        从alias table中采样的时间复杂度为O(1), 构建alias table的时间复杂度为O(n).

        Args:
            area_ratio: 所有id被采样的概率值，已归一化;
        """
        l = len(area_ratio)
        accept, alias = [0]*l, [0]*l
        small, large = [], []
        area_ratio_ = np.array(area_ratio)*l

        for i, prob in enumerate(area_ratio_):
            if (prob < 1.):
                small.append(i)
            else:
                large.append(i)

        while(small and large):
            small_idx, large_idx = small.pop(), large.pop()
            accept[small_idx] = area_ratio_[small_idx]
            alias[small_idx] = large_idx

            # 把大的概率多的部分拿出来填小的
            area_ratio_[large_idx] = area_ratio_[
                large_idx] - (1-area_ratio[small_idx])

            # 每一次while循环必然会处理一个small或large
            if(area_ratio_[large_idx] < 1.):
                small.append(large_idx)
            else:
                large.append(large_idx)

        return accept, alias

    def get_alias_edge(self, pre, cur):
        """获取采样表
        Args:
            pre: string, 前一个结点的名称;
            cur: string, 当前结点名称;

        """
        unnormlized_probs = []
        next_items = []
        # 构建alias table需要先排序
        next_dist = sorted(
            self.transfer_matrix[cur].items(), key=lambda x: x[1], reverse=True)
        for x, weight in next_dist:
            next_items.append(x)
            if(x == pre):  # 往回走, alpha = x/p
                unnormlized_probs.append(weight/self.p)
            elif (x in self.transfer_matrix[pre]):  # BFS
                unnormlized_probs.append(weight)
            else:  # DFS, alpha = x/q
                unnormlized_probs.append(weight/self.q)

        norm_const = sum(unnormlized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormlized_probs]

        accept, alias = self.create_alias_table(normalized_probs)
        return accept, alias, next_items

    def alias_sample(self, accept, alias):
        """alias采样
        """
        N = len(accept)
        # 先采样一个索引
        i = int(np.random.uniform()*N)
        # 再随机一个概率
        r = np.random.rand()

        # 比较随机概率与索引概率的大小
        if (r < accept[i]):
            return i
        else:
            return alias[i]

    def _one_walk(self, first_item):
        sample = [first_item]
        # pick the first element
        cur_item = first_item
        pre_item = ""

        cnt = 1
        while (cnt < self.sample_length):
            if (cur_item not in self.transfer_matrix) or (cur_item not in self.item_distributions):
                break

            if (cnt == 1):  # 在选择第一个结点的下一个结点时，直接根据相邻节点的概率值分布来进行采样
                next_distributions = self.transfer_matrix[cur_item]
                # 排序
                next_dist = sorted(next_distributions.items(),
                                   key=lambda x: x[1], reverse=True)
                next_items, next_probs = [], []
                for item, weight in next_dist:
                    next_items.append(item)
                    next_probs.append(weight)
                # 创建alias table
                accept, alias = self.create_alias_table(next_probs)
                # alias sample
                sample_idx = self.alias_sample(accept, alias)
                # 更新cur_item, pre_item和采样表
                pre_item = cur_item
                cur_item = next_items[sample_idx]
                sample.append(cur_item)
            else:
                # 先判断之前是否计算过(pre_item, cur_item)对的alias table，若有则可以直接提取信息而不必重新生成
                if((pre_item, cur_item) in self.alias_edges):
                    accept, alias, next_items = self.alias_edges[(
                        pre_item, cur_item)]
                else:  # 未计算过，重新计算
                    accept, alias, next_items = self.get_alias_edge(
                        pre_item, cur_item)
                    # 更新alias_table 键
                    self.alias_edges[(pre_item, cur_item)] = (
                        accept, alias, next_items)

                # 采样
                sample_idx = self.alias_sample(accept, alias)
                # 更新cur_item, pre_item和采样表
                pre_item = cur_item
                cur_item = next_items[sample_idx]
                sample.append(cur_item)

            cnt += 1

        return sample
