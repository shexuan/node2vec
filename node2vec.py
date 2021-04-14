from collections import defaultdict
import random
from pyspark.ml.feature import CountVectorizer, Word2Vec, Word2VecModel


def generate_transfer_matrix(df, column='camp_id'):
	"""利用序列来生成概率转移矩阵
	Args:
		df: spark dataframe;
		column: 字段名称，用来生成概率转移矩阵的字段;
	"""
    # 把每一对相邻的word整理成一个可以转移的word pair
    # 注：需要先过滤一下原始序列, 否则如果只有一个元素的话，这里会报错
    pair_udf = udf(lambda lst: [[lst[i-1], lst[i]]
                                for i in range(1, len(lst))], ArrayType(ArrayType(StringType())))

    # 计算每一个转移对出现的次数
    pair_counts = df.withColumn("pairs", pair_udf(column))\
                    .withColumn("pair", Func.explode("pairs"))\
                    .groupby("pair").count()\
                    .select('pair', 'count').collect()
    print('Total pair numbers:', len(pair_counts))

    # 使用嵌套哈希表存储概率转移矩阵
    transfer_matrix = defaultdict(dict)
    # 所有pair起点item的出现次数统计，用于选择生成sentence时选择第一个id
    item_count = defaultdict(int)

    tot_start = 0.
    for row in pair_counts:
        transfer_matrix[row['pair'][0]][row['pair'][1]] = row['count']
        item_count[row['pair'][0]] += row['count']
        tot_start += row['count']

    # 对每个起始点计算归一化的转移概率
    for key, dic in transfer_matrix.items():
        tot = 0.
        # 先计算总和
        for val, cnt in dic.items():
            tot += cnt
        # 归一化
        for val, cnt in dic.items():
            transfer_matrix[key][val] = cnt/float(tot)

    # start item in pair distribution
    item_distributions = dict()
    for item, cunt in item_count.items():
        item_distributions[item] = cunt/float(tot_start)

    return transfer_matrix, item_distributions


#########################################################################
##################### Deepwalk ##########################################

def _one_walk(transfer_matrix, item_distributions, sample_length):
    """使用轮盘法来选择每个item.
    Args:
        transfer_matrix: 嵌套字典，每个item的下一个item的概率分布，已归一化；
        item_distributions: 字典，原始所有pair的起始item的概率分布，已归一化；
        sample_length: int，每次采样长度；
    """
    sample = []
    # pick the first element
    prob_first = random.random()
    cur_item = ""
    accumulate_prob = 0.
    for item, prob in item_distributions.items():
        accumulate_prob += prob
        if (accumulate_prob >= prob_first):
            cur_item = item
            sample.append(item)
            break

    cnt = 1
    while (cnt <= sample_length):
        if (cur_item not in transfer_matrix) or (cur_item not in item_distributions):
            break
        next_distributions = transfer_matrix[cur_item]
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


def random_walk(transfer_matrix, item_distributions, sample_length, sample_num):
    """ 随机游走，利用轮盘法来进行采样每个id
    Args:
        transfer_matrix: 嵌套字典，每个item的下一个item的概率分布，已归一化；
        item_distributions: 字典，原始所有pair的起始item的概率分布，已归一化；
        sample_length: int，每次采样长度；
        sample_num: int, 采样生成的sentence数量；
    """
    samples = []
    for i in range(sample_num):
        samples.append(_one_walk(transfer_matrix,
                                 item_distributions, sample_length))

    return samples


def graph_emb(samples, w2v_model_save_path=None, w2v_args=None):
    """利用随机游走采样结果来进行word2vec训练
    Args:
        samples: list, 随机游走的采样结果；
        w2v_model_save_path: word2vec模型的保存路径；
        w2v_args: dict, word2vec模型的训练参数,  
            e.g., {"vectorSize":64, "minCount":3, "seed":123, "numPartitions":64, "maxIter":5, "maxSentenceLength":128, "windowSize":5} or vectorSize=64, minCount=3, ...}
    """
    data = sc.parallelize(samples).toDF()
    columns = ['_'+str(i) for i in range(1, len(samples[0])+1)]
    data = data.withColumn('sentence', Func.array(columns)).select('sentence')

    w2v = Word2Vec(inputCol='sentence', outputCol='vec', **w2v_args)
    w2ver = w2v.fit(data)
    # data = w2ver.transform(data)

    if w2v_model_save_path:
        # save model
        w2ver.write().overwrite().save(w2v_model_save_path)
        # load model
        # w2v_model = Word2VecModel.load(model_path)

    return w2ver


#########################################################################
##################### Node2vec ##########################################


def create_alias_table(area_ratio):
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


def alias_sample(accept, alias):
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


def get_alias_edge(G, pre, cur, p, q):
    """获取采样表
    Args:
        G: dict，概率转移矩阵，{'n1':{'n2':0.1, 'n3':0.2, 'n4':0.7}, ...};
        pre: string, 前一个结点的名称;
        cur: string, 当前结点名称;
        p: float, 往回走的系数;
        q: float, DFS的系数, BFS的系数为1;
    """
    unnormlized_probs = []
    next_items = []
    # 构建alias table需要先排序
    next_dist = sorted(G[cur].items(), key=lambda x: x[1], reverse=True)
    for x, weight in next_dist:
        next_items.append(x)
        if(x == pre):  # 往回走, alpha = x/p
            unnormlized_probs.append(weight/p)
        elif (x in G[pre]):  # BFS
            unnormlized_probs.append(weight)
        else:
            unnormlized_probs.append(weight/q)

    norm_const = sum(unnormlized_probs)
    normalized_probs = [
        float(u_prob)/norm_const for u_prob in unnormlized_probs]

    accept, alias = create_alias_table(normalized_probs)
    return accept, alias, next_items


def _node2vec_one_walk(transfer_matrix, item_distributions, alias_edges, sample_length=32, p=4, q=0.25):
    """一次walk
    Args:
        transfer_matrix: dict，概率转移矩阵，{'n1':{'n2':0.1, 'n3':0.2, 'n4':0.7}, ...};
        item_distributions: dict, 已归一化的所有结点出现频率;
        sample_length: int，每次采样长度;
        p: float, 往回走的系数;
        q: float, DFS的系数, BFS的系数为1;
        alias_edges:dict, 储存着alias采样表信息，key = (pre_item, cur_item), value = (accept_table, alias_table, next_items),
            每次遇到新的(pre_item, cur_item)则更新.
    """
    sample = []
    # pick the first element
    prob_first = random.random()
    cur_item = ""
    pre_item = ""
    accumulate_prob = 0.
    for item, prob in item_distributions.items():
        accumulate_prob += prob
        if (accumulate_prob >= prob_first):
            cur_item = item
            sample.append(item)
            break

    cnt = 1
    while (cnt < sample_length):
        if (cur_item not in transfer_matrix) or (cur_item not in item_distributions):
            break

        if (cnt == 1):  # 在选择第一个结点的下一个结点时，直接根据相邻节点的概率值分布来进行采样
            next_distributions = transfer_matrix[cur_item]
            # 排序
            next_dist = sorted(next_distributions.items(),
                               key=lambda x: x[1], reverse=True)
            next_items, next_probs = [], []
            for item, weight in next_dist:
                next_items.append(item)
                next_probs.append(weight)
            # 创建alias table
            accept, alias = create_alias_table(next_probs)
            # alias sample
            sample_idx = alias_sample(accept, alias)
            # 更新cur_item, pre_item和采样表
            pre_item = cur_item
            cur_item = next_items[sample_idx]
            sample.append(cur_item)
        else:
            # 先判断之前是否计算过(pre_item, cur_item)对的alias table，若有则可以直接提取信息而不必重新生成
            if((pre_item, cur_item) in alias_edges):
                accept, alias, next_items = alias_edges[(pre_item, cur_item)]
            else:  # 未计算过，重新计算
                accept, alias, next_items = get_alias_edge(
                    transfer_matrix, pre_item, cur_item, p, q)
                # 更新alias_table 键
                alias_edges[(pre_item, cur_item)] = (accept, alias, next_items)

            # 采样
            sample_idx = alias_sample(accept, alias)
            # 更新cur_item, pre_item和采样表
            pre_item = cur_item
            cur_item = next_items[sample_idx]
            sample.append(cur_item)

        cnt += 1

    return sample
