import pandas as pd
import numpy as np

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
    def __init__(self, TargetSim: pd.DataFrame, ContentSim: pd.DataFrame, top_n: int,
                 order: pd.DataFrame, target_col: str, content_col: str, Popularity = None,
                 user_group_type: pd.DataFrame = None):
        self.TargetSim = TargetSim
        self.ContentSim = ContentSim
        self.Pop = Popularity
        self.O = order
        # self.S_keys = sim.columns.values.astype('int')
        self.N = top_n
        self.L = len(TargetSim)
        self.T = target_col
        self.C = content_col
        self.UG = user_group_type

    def _get_targets_top_n_corr_targets(self):
        targets = np.unique(self.TargetSim['target_id'].values)
        top_n_corr_targets = []
        for t in targets:
            top_n_corr_targets.append(self.TargetSim[self.TargetSim['target_id'] == t]['corr_target_id'].tolist())
        return targets, top_n_corr_targets

    def _get_corr_targets_contents(self, targets, current_contents):
        other_contents = []
        for t in targets:
            other_contents += self.O[self.O[self.T] == t][self.C].tolist()
        other_contents = list(set(other_contents))
        return [i for i in other_contents if i not in current_contents]

    def _get_curr_contents_similar_contents(self, current_contents):
        similar_contents = []
        for c in current_contents:
            similar_contents += self.ContentSim[self.ContentSim['target_id'] == c]['corr_target_id'].tolist()
        return [i for i in similar_contents if i not in current_contents]

    def _get_recommend_contents(self):
        targets, top_n_corr_targets = self._get_targets_top_n_corr_targets()
        recommends = []

        for i in range(len(targets)):
            curr_target = targets[i]
            curr_contents = self._get_corr_targets_contents([curr_target], [])
            recommend_contents = self._get_corr_targets_contents(top_n_corr_targets[i], curr_contents) + \
                                 self._get_curr_contents_similar_contents(curr_contents)
            top_n_contents = recommend_contents[:self.N] if len(recommend_contents) > self.N else recommend_contents
            if len(top_n_contents) < self.N:
                if self.Pop is not None:
                    target_type = self.UG[self.UG[self.T] == curr_target]['concat_grp_type'].values
                    if len(target_type) > 0:
                        pop_list = self.Pop[self.Pop['concat_grp_type'] == target_type[0]]['popular_item'].tolist()
                        top_n_contents += pop_list[:self.N - len(top_n_contents)] if pop_list else [None] * (
                                self.N - len(top_n_contents))
            recommends.append(top_n_contents)
        self.recommends = recommends
        self.targets = targets

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
            return None
