from typing import Tuple

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import math
from pipeline import ProcessingPipeline


class Similarity:
    def __init__(self, mode: str, user_info: pd.DataFrame = None, item_info: pd.DataFrame = None,
                 order: pd.DataFrame = None, target_col: str = None, content_col: str = None, top_n: int = None):

        """
        similarity table
        -------------------------
        target_id        | int
        -------------------------
        corr_target_id   | int
        -------------------------
        similarity       | float
        -------------------------`

        :param mode:
            {
            'UL': user label similarity,
            'IL': item label similarity,
            'UC': user cluster similarity,
            'IC': item cluster similarity,
            'ALL': all similarity
            }
        :param user_info:
        :param item_info:
        :param order:
        :param target_col: used for cluster similarity calculation, cluster by target_col
        :param content_col: used for cluster similarity calculation,  cluster on content_col
        :param top_n: top_n similarity will be maintain
        """

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
        data = np.zeros(shape = (len(self.Sim_T) * self.N, 3,))
        for i in range(len(self.Sim_T)):
            t = np.array([self.Sim_T[i] for _ in range(self.N)])
            k = self.Sim_K[i]
            v = self.Sim[i]
            _tmp = np.array([t, k, v]).T
            data[i * self.N: (i + 1) * self.N] = _tmp
        data = pd.DataFrame(data = data, columns = ['target_id', 'corr_target_id', 'similarity'])
        return data

    @staticmethod
    def _eculid_distance_old_version(v1: np.array, v2: np.array) -> np.array:
        '''
        calculate eculid_distance between two arrry.
        :param v1: (1) array
        :param v2: (n) array
        :return: (n,1) array
        '''

        return np.linalg.norm(v1 - v2, axis = 1, keepdims = True)

    @staticmethod
    def _cosine_distance_old_version(content1, content2) -> float:
        '''
        simple method for calculate cosine distance.
        e.g: x = [1 0 1 1 0], y = [0 1 1 0 1]
             cosine = (x1*y1+x2*y2+...) / [sqrt(x1^2+x2^2+...)+sqrt(y1^2+y2^2+...)]
             that means union_len(items1, items2) / sqrt(len(items1)*len(items2))
        '''
        if set(content1) == set(content2):
            return 0.0

        union_len = len(set(content1) & set(content2))
        if union_len == 0:
            return 0.0
        cosine = union_len / math.sqrt(len(content1) * len(content2))
        return cosine

    @staticmethod
    def _get_squared_euclidean_distances(A: np.array, B: np.array) -> np.array:
        BT = B.transpose()
        vec_product = np.dot(A, BT)
        sq_A = A ** 2
        sum_sq_A = np.matrix(np.sum(sq_A, axis = 1))
        sum_sq_A_ex = np.tile(sum_sq_A.transpose(), (1, vec_product.shape[1]))
        sq_B = B ** 2
        sum_sq_B = np.sum(sq_B, axis = 1)
        sum_sq_B_ex = np.tile(sum_sq_B, (vec_product.shape[0], 1))
        sq_euclid_dis = sum_sq_A_ex + sum_sq_B_ex - 2 * vec_product
        sq_euclid_dis[sq_euclid_dis < 0] = 0.0
        return np.matrix(sq_euclid_dis)

    @staticmethod
    def _cosine_distance(A: np.array, B: np.array) -> float:
        return cosine_similarity(A, B)

    def _label_euclid_similarity(self, data: pd.DataFrame, target_col) -> Tuple[np.array, np.array, np.array]:
        '''
        calculate euclid_similarity between vectors, in this case vectors are labels
        '''
        print('calculating euclid_similarity...')
        targets = data[target_col].values
        labels = data.loc[:, data.columns != target_col].values
        labels = preprocessing.StandardScaler().fit_transform(labels)
        labels[np.isnan(labels)] = 0

        similarity = self._get_squared_euclidean_distances(labels, labels)
        idx_is_zeros = np.where(similarity == 0)
        similarity[idx_is_zeros] = np.inf
        ind = np.argsort(similarity, axis = 1)[:, :self.N]
        top_similarity = np.take_along_axis(similarity, ind, axis = 1)
        top_other_targets = targets[ind]

        idx_include = np.where(top_similarity != np.inf)
        _range = np.max(top_similarity[idx_include]) - np.min(top_similarity[idx_include])
        top_similarity[idx_include] = 1 - ((top_similarity[idx_include] - np.min(top_similarity[idx_include])) / _range)
        top_similarity[np.where(top_similarity == np.inf)] = 0
        print('Finished calculating euclid_similarity')
        return top_similarity.getA(), top_other_targets, targets

    def _cluster_cosine_similarity(self, data: pd.DataFrame, target_col, content_col):
        print('Calculating cosine_similarity...')
        _tmp = pd.DataFrame([1] * len(data), index = [data[target_col].tolist(), data[content_col].tolist()])
        _tmp = _tmp[~_tmp.index.duplicated(keep = 'first')]
        _tmp = _tmp.unstack()
        _tmp = _tmp.fillna(0)
        targets = _tmp.index.values

        batches = ProcessingPipeline(targets, max_n = 1000).get_batches()
        all_top_similarity = None
        all_top_other_targets = None
        for (i, j) in batches:
            _compare_tmp = _tmp[i:j]
            similarity = self._cosine_distance(_compare_tmp, _tmp.values)
            similarity = np.where(similarity > 0.99, 0, similarity)
            ind = np.argsort(similarity, axis = 1)[:, -self.N:][::-1]

            top_similarity = np.take_along_axis(similarity, ind, axis = 1)
            top_other_targets = targets[ind]
            if all_top_similarity is None:
                all_top_similarity = top_similarity
                all_top_other_targets = top_other_targets
                continue

            all_top_similarity = np.vstack((all_top_similarity, top_similarity))
            all_top_other_targets = np.vstack((all_top_similarity, top_other_targets))
        print('Finished calculating cosine_similarity')
        return np.array(all_top_similarity), np.array(all_top_other_targets), targets

    def _get_similarity(self):
        if self.M == 'UL':
            self.Sim, self.Sim_K, self.Sim_T = self._label_euclid_similarity(self.U, self.T)
        if self.M == 'IL':
            self.Sim, self.Sim_K, self.Sim_T = self._label_euclid_similarity(self.I, self.C)
        if self.M == 'UC':
            self.Sim, self.Sim_K, self.Sim_T = self._cluster_cosine_similarity(self.O, self.T, self.C)
        if self.M == 'IC':
            self.Sim, self.Sim_K, self.Sim_T = self._cluster_cosine_similarity(self.O, self.C, self.T)
