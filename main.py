from collections import defaultdict
import random
from pyspark.ml.feature import CountVectorizer, Word2Vec, Word2VecModel
import numpy as np
from node2vec import DeepWalk, Node2vec
from gensim.models import word2vec

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, datefmt='%m/%d/%Y %H:%M:%S')


def generate_transfer_matrix(df, column='camp_id'):
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


def graph_emb(self, samples, w2v_model_save_path=None, w2v_args=None):
    """利用随机游走采样结果来进行word2vec训练, 这里用的是spark的word2vec。
    也可以改为local版的gensim进行训练
    Args:
        samples: list, 随机游走的采样结果；
        w2v_model_save_path: word2vec模型的保存路径；
        w2v_args: dict, word2vec模型的训练参数,  
            e.g., {"vectorSize":64, "minCount":3, "seed":123, "numPartitions":64, "maxIter":5, "maxSentenceLength":128, "windowSize":5} or vectorSize=64, minCount=3, ...}
    """
    data = sc.parallelize(samples).toDF()
    columns = ['_'+str(i) for i in range(1, len(samples[0])+1)]
    data = data.withColumn(
        'sentence', Func.array(columns)).select('sentence')

    w2v = Word2Vec(inputCol='sentence', outputCol='vec', **w2v_args)
    w2ver = w2v.fit(data)
    # data = w2ver.transform(data)

    if w2v_model_save_path:
        # save model
        w2ver.write().overwrite().save(w2v_model_save_path)
        # load model
        # w2v_model = Word2VecModel.load(model_path)

    return w2ver


def graph_emb_local(sentences, w2v_model_save_path, w2v_args):
    # 利用随机游走采样结果来进行word2vec训练
    # w2v_args: dict, word2vec模型的训练参数,
    #  e.g., {"size":128, "window":5, "min_count":1, "sg":1, "workers":16, "iter":10}
    model = word2vec.Word2Vec(deepwalks, **w2v_args)

    base_dir = os.path.dirname(w2v_model_save_path)
    if base_dir and (not os.path.exists(base_dir)):
        os.makedirs(base_dir)

    print("save w2v to {}".format(w2v_model_save_path))
    pickle.dump(model, open(w2v_model_save_path, 'wb'))

    # load model
    # model = pickle.load(open(w2v_model_save_path, 'rb'))
    return model


def main():
    # 读取文件
    df = spark.read.parquet(fpath)

    # 生成概率转移矩阵
    transfer_matrix, item_distributions = generate_transfer_matrix(
        df, column='camp')

    # deepwalk
    DW = DeepWalk(transfer_matrix, item_distributions, 64, 1000000)
    deepwalks = DW.simulate_walks()

    # node2vec
    N2v = Node2vec(transfer_matrix, item_distributions,
                   64, 1000000, p=0.25, q=4)
    node2vecs = N2v.simulate_walks()

    # 利用word2vec来进行embedding
    args = {"vectorSize": 64, "minCount": 3, "seed": 123, "numPartitions": 64,
            "maxIter": 5, "maxSentenceLength": 128, "windowSize": 5}
    w2ver = graph_emb(node2vecs, w2v_args=args)

    return w2ver


if __name__ == '__main__':
    main()
