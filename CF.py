import pandas as pd

'''
################################################## CF calculation ######################################################
recommend results
-------------------------
target_id           | int
-------------------------
recommend_content_1 | int 
-------------------------
....                | int 
-------------------------
recommend_content_n | int 
-------------------------
'''


class CF:
    def __init__(self, targets, TargetSim: pd.DataFrame, ContentSim: pd.DataFrame, top_n: int,
                 order: pd.DataFrame, target_col: str, content_col: str, Popularity=None,
                 user_group_type=None):
        '''
        :param targets: 协同过滤的目标，
            如果目标是用户,该项应为用户的数据，结果则计算为用户推荐协同过滤产品；
            如果目标是产品，该项则为产品数据，结果则为产品推荐协同过滤的用户
        :param TargetSim: 目标的相似度
        :param ContentSim: 过滤内容的相似度。过滤内容为要推荐的内容。如向用户推荐产品，则应为产品的相似度
        :param top_n: top_n for saving
        :param order: 订购历史
        :param target_col: target id col 的 名称
        :param content_col: content id  col 的 名称
        :param Popularity: 流行度的列表
        :param user_group_type: 如果user需要group by 某一特征，则写入该特征
        '''
        self.targets = targets
        self.TargetSim = TargetSim
        self.ContentSim = ContentSim
        self.Pop = Popularity
        self.O = order
        # self.S_keys = sim.columns.values.astype('int')
        self.N = top_n
        self.L = len(TargetSim) if TargetSim is not None else 0
        self.T = target_col
        self.C = content_col
        self.UG = user_group_type

    def get_recommend_contents_df(self):
        print('getting recommend_contents_df')
        self._get_recommend_contents()
        if self.recommends:
            cols = []
            for i in range(1, self.N + 1):
                cols.append(f'recommend_content_{i}')
            df = pd.DataFrame(self.recommends)
            df['target_id '] = self.targets
            return df
        else:
            return

    def _get_targets_top_n_corr_targets(self):
        if self.targets is None:
            return None
        top_n_corr_targets = []
        for t in self.targets:
            top_n_corr_targets.append(self.TargetSim[self.TargetSim['target_id'] == t]['corr_target_id'].tolist())
        return top_n_corr_targets

    def _get_corr_targets_contents(self, targets, current_contents):
        if self.targets is None or current_contents is None:
            return None
        other_contents = []
        other_contents += self.O[self.O[self.T].isin(targets)][self.C].tolist()
        return list(set(other_contents) ^ set(current_contents))

    def _get_curr_contents_similar_contents(self, current_contents):
        if current_contents is None:
            return None
        similar_contents = []
        similar_contents += self.ContentSim[self.ContentSim['target_id'].isin(current_contents)][
            'corr_target_id'].tolist()
        return list(set(similar_contents) ^ set(current_contents))

    def _get_recommend_contents(self):
        if self.targets is None or self.O is None:
            self.recommends = None
            return

        top_n_corr_targets = self._get_targets_top_n_corr_targets()
        recommends = []
        pop_list = []
        if self.Pop is not None:
            pop_list = self.Pop[self.Pop['concat_grp_type'] == self.UG]['popular_item'].tolist()

        for i in range(len(self.targets)):
            curr_target = self.targets[i]
            curr_contents = self._get_corr_targets_contents([curr_target], [])

            recommend_contents = self._get_corr_targets_contents(top_n_corr_targets[i], curr_contents) + \
                                 self._get_curr_contents_similar_contents(curr_contents)

            top_n_contents = recommend_contents[:self.N] if len(recommend_contents) > self.N else recommend_contents
            if len(top_n_contents) < self.N:
                top_n_contents += pop_list[:self.N - len(top_n_contents)] if pop_list else [None] * (
                        self.N - len(top_n_contents))
            recommends.append(top_n_contents)
        self.recommends = recommends
        return
