import math
from pipeline import ProcessingPipeline
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

'''
############################################# similarity calculation ###################################################
similarity table
-------------------------
target_id        | int
-------------------------
corr_target_id   | int 
-------------------------
similarity       | float
-------------------------
'''


class Similarity:
    def __init__(self, mode: str, user_info: pd.DataFrame = None, item_info: pd.DataFrame = None,
                 order: pd.DataFrame = None, target_col: str = None, content_col: str = None, top_n: int = None):
        '''
        :param mode:
            {
            'UL': user label similarity, 用户标签相似度
            'IL': item label similarity, 产品标签相似度
            'UC': user cluster similarity, 用户行为相似度
            'IC': item cluster similarity, 产品覆盖用户相似度
            'ALL': all similarity 全部
            }
        :param user_info: user data 用户数据
        :param item_info: item data 产品数据（或其他）
        :param order: order history data 订单历史数据
        :param target_col: used for cluster similarity calculation, cluster by target_col 目标id的col
        :param content_col: used for cluster similarity calculation,  cluster on content_col 内容id的col
        :param top_n: top_n similarity will be maintain 储存topn的结果
        '''

        self.U = user_info
        self.I = item_info
        self.O = order
        self.M = mode
        self.T = target_col
        self.C = content_col
        self.N = top_n

    def get_similarity_df(self) -> pd.DataFrame:
        '''
        save similarity to csv
        '''
        self._get_similarity()
        if self.N == 0 or self.Sim_T is None:
            return pd.DataFrame(data=None, columns=['target_id', 'corr_target_id', 'similarity'])
        data = np.zeros(shape=(len(self.Sim_T) * self.N, 3,))
        for i in range(len(self.Sim_T)):
            t = np.array([self.Sim_T[i] for _ in range(self.N)])
            k = self.Sim_K[i]
            v = self.Sim[i]
            _tmp = np.array([t, k, v]).T
            data[i * self.N: (i + 1) * self.N] = _tmp
        data = pd.DataFrame(data=data, columns=['target_id', 'corr_target_id', 'similarity'])
        return data

    @staticmethod
    def _eculid_distance(v1: np.array, v2: np.array) -> np.array:
        '''
        calculate eculid_distance between two arrry.
        :param v1: (1) array
        :param v2: (n) array
        :return: (n,1) array
        '''

        return np.linalg.norm(v1 - v2, axis=1, keepdims=True)

    @staticmethod
    def _cosine_distance(v1: np.array, v2: np.array) -> float:
        '''
        simple method for calculate cosine distance.
        e.g: x = [1 0 1 1 0], y = [0 1 1 0 1]
             cosine = (x1*y1+x2*y2+...) / [sqrt(x1^2+x2^2+...)+sqrt(y1^2+y2^2+...)]
             that means union_len(items1, items2) / sqrt(len(items1)*len(items2))
        '''
        if set(v1) == set(v2):
            return 0.0

        union_len = len(set(v1) & set(v2))
        if union_len == 0:
            return 0.0
        cosine = union_len / math.sqrt(len(v1) * len(v2))
        return cosine

    @staticmethod
    def _get_squared_euclidean_distances(A: np.array, B: np.array) -> np.array:
        '''
        calculate euclidean distances between two set of vectors
        :param A: 矩阵
        :param B: 矩阵
        :return:
        '''
        BT = B.transpose()
        vec_product = np.dot(A, BT)
        sq_A = A ** 2
        sum_sq_A = np.matrix(np.sum(sq_A, axis=1))
        sum_sq_A_ex = np.tile(sum_sq_A.transpose(), (1, vec_product.shape[1]))
        sq_B = B ** 2
        sum_sq_B = np.sum(sq_B, axis=1)
        sum_sq_B_ex = np.tile(sum_sq_B, (vec_product.shape[0], 1))
        sq_euclid_dis = sum_sq_A_ex + sum_sq_B_ex - 2 * vec_product
        sq_euclid_dis[sq_euclid_dis < 0] = 0.0
        return np.matrix(sq_euclid_dis)

    @staticmethod
    def _get_cosine_distance(A: np.array, B: np.array) -> float:
        return cosine_similarity(A, B)

    def _label_euclid_similarity(self, data: pd.DataFrame, target_col):
        '''
        calculate euclid_similarity between vectors, in this case vectors are labels
        '''
        print('calculating euclid_similarity...')
        targets = data[target_col].values
        labels = data.loc[:, data.columns != target_col].values
        labels = preprocessing.StandardScaler().fit_transform(labels)
        labels[np.isnan(labels)] = 0

        batches = ProcessingPipeline(targets, max_n=1000).get_batches()
        all_top_similarity = None
        all_top_other_targets = None
        for (i, j) in batches:
            compare_labels = labels[i:j]
            similarity = self._get_squared_euclidean_distances(compare_labels, labels)
            idx_is_zeros = np.where(similarity == 0)
            similarity[idx_is_zeros] = np.inf
            ind = np.argsort(similarity, axis=1)[:, :self.N]
            top_similarity = np.take_along_axis(similarity, ind, axis=1)
            top_other_targets = targets[ind]
            idx_include = np.where(top_similarity != np.inf)
            _range = np.max(top_similarity[idx_include]) - np.min(top_similarity[idx_include])
            top_similarity[idx_include] = 1 - (
                    (top_similarity[idx_include] - np.min(top_similarity[idx_include])) / _range)
            top_similarity[np.where(top_similarity == np.inf)] = 0

            if all_top_similarity is None:
                all_top_similarity = top_similarity
                all_top_other_targets = top_other_targets
                continue

            all_top_similarity = np.vstack((all_top_similarity, top_similarity))
            all_top_other_targets = np.vstack((all_top_other_targets, top_other_targets))

        print('Finished calculating euclid_similarity')
        return np.array(all_top_similarity), np.array(all_top_other_targets), targets

    def _cluster_cosine_similarity(self, data: pd.DataFrame, target_col, content_col):

        print('Calculating cosine_similarity...')
        _tmp = pd.DataFrame([1] * len(data), index=[data[target_col].tolist(), data[content_col].tolist()])
        _tmp = _tmp[~_tmp.index.duplicated(keep='first')]
        _tmp = _tmp.unstack()
        _tmp = _tmp.fillna(0)
        targets = _tmp.index.values
        tmp = _tmp.values

        batches = ProcessingPipeline(targets, max_n=1000).get_batches()
        all_top_similarity = None
        all_top_other_targets = None
        for (i, j) in batches:
            _compare_tmp = tmp[i:j]
            similarity = self._get_cosine_distance(_compare_tmp, tmp)
            similarity = np.where(similarity > 0.99, 0, similarity)
            ind = np.argsort(similarity, axis=1)[:, -self.N:][::-1]
            top_similarity = np.take_along_axis(similarity, ind, axis=1)
            top_other_targets = targets[ind]
            if all_top_similarity is None:
                all_top_similarity = top_similarity
                all_top_other_targets = top_other_targets
                continue

            all_top_similarity = np.vstack((all_top_similarity, top_similarity))
            all_top_other_targets = np.vstack((all_top_other_targets, top_other_targets))
        print('Finished calculating cosine_similarity')
        return np.array(all_top_similarity), np.array(all_top_other_targets), targets

    def _group_label_euclid_similarity(self, data: pd.DataFrame, target_col, group_col):
        print('Calculating group euclid_similarity...')
        targets = data[group_col].values
        _data = data.drop(target_col, axis=1, inplace=False)
        _group_label_data = pd.DataFrame()
        for g, d in _data.groupby(group_col):
            mean = d.drop(group_col, axis=1, inplace=False)
            _group_label_data = _group_label_data.append(mean)

        labels = _group_label_data.values
        labels = preprocessing.StandardScaler().fit_transform(labels)
        labels[np.isnan(labels)] = 0
        similarity = self._get_squared_euclidean_distances(labels, labels)

        idx_is_zeros = np.where(similarity == 0)
        similarity[idx_is_zeros] = np.inf
        ind = np.argsort(similarity, axis=1)[:, :self.N]
        top_similarity = np.take_along_axis(similarity, ind, axis=1)
        top_other_targets = targets[ind]

        idx_include = np.where(top_similarity != np.inf)
        _range = np.max(top_similarity[idx_include]) - np.min(top_similarity[idx_include])
        top_similarity[idx_include] = 1 - ((top_similarity[idx_include] - np.min(top_similarity[idx_include])) / _range)
        top_similarity[np.where(top_similarity == np.inf)] = 0
        print('Finished calculating group euclid_similarity')
        return top_similarity.getA(), top_other_targets, targets

    def _knn_similarity_minkowski(self, data: pd.DataFrame, target_col):
        if self.N == 0 or data is None:
            return None, None, None

        targets = data[target_col].values
        labels = data.loc[:, data.columns != target_col].values
        neigh = NearestNeighbors(n_neighbors=self.N, radius=0.4, n_jobs=-1, leaf_size=10)
        neigh.fit(labels)
        knn_res = neigh.kneighbors(labels, self.N, return_distance=True)
        all_top_similarity = preprocessing.MinMaxScaler().fit_transform(knn_res[0])
        ind = knn_res[1]
        all_top_other_targets = targets[ind]
        return all_top_similarity, all_top_other_targets, targets

    def _knn_similarity_jaccard(self, data: pd.DataFrame, target_col, content_col):
        if self.N == 0 or data is None or len(data) < 1:
            return None, None, None
        _tmp = pd.DataFrame([1] * len(data), index=[data[target_col].tolist(), data[content_col].tolist()])
        _tmp = _tmp[~_tmp.index.duplicated(keep='first')]
        _tmp = _tmp.unstack()
        _tmp = _tmp.fillna(0)
        targets = _tmp.index.values
        tmp = _tmp.values
        neigh = NearestNeighbors(n_neighbors=self.N, radius=0.4, n_jobs=-1, leaf_size=10)
        neigh.fit(tmp)
        knn_res = neigh.kneighbors(tmp, self.N, return_distance=True)
        all_top_similarity = preprocessing.MinMaxScaler().fit_transform(knn_res[0])
        ind = knn_res[1]
        all_top_other_targets = targets[ind]
        return all_top_similarity, all_top_other_targets, targets

    def _get_similarity(self):
        if self.M == 'UL':
            self.Sim, self.Sim_K, self.Sim_T = self._knn_similarity_minkowski(self.U, self.T)
        if self.M == 'IL':
            self.Sim, self.Sim_K, self.Sim_T = self._knn_similarity_minkowski(self.I, self.T)
        if self.M == 'UC':
            self.Sim, self.Sim_K, self.Sim_T = self._knn_similarity_jaccard(self.O, self.T, self.C)
        if self.M == 'IC':
            self.Sim, self.Sim_K, self.Sim_T = self._knn_similarity_jaccard(self.O, self.T, self.C)
        if self.M == 'GL':
            self.Sim, self.Sim_K, self.Sim_T = self._group_label_euclid_similarity(self.U, self.T, 'grp_ind_type_1')
