import pandas as pd
from datetime import datetime

'''
################################################## data processing #####################################################
'''


class DataProcessing:
    def __init__(self, data, drop_col: list, dummy_col: list, time_col: list
                 , rename_pair: dict, str2num_col: list,
                 id_col: str):
        self.D = data
        self.drop_col = drop_col
        self.dummy_col = dummy_col
        self.time_col = time_col
        self.rename_pair = rename_pair
        self.str2num_col = str2num_col
        self.id_col = id_col
        self.protect_col = ['grp_ind_type_1']  # 'concat_grp_type',

        self._get_dummy_col()
        self._get_time_long_col()
        self._refine_col_name()
        self._str_col_to_num()
        self._drop_col()
        self._df_final_refine()

    def _get_dummy_col(self):
        print(f'contructing dummy cols')
        dummy_df = pd.DataFrame()
        for i in self.dummy_col:
            tmp_df = pd.get_dummies(self.D[i], prefix = i)
            dummy_df = pd.concat([dummy_df, tmp_df], axis = 1)
        self.D = pd.concat([self.D, dummy_df], axis = 1)
        print(f'contructed {len(dummy_df.columns)} dummy cols')

    def _get_time_long_col(self):

        def _calculate_time_long(base_date, date):
            return (date - base_date).days

        base_date = datetime.now()
        print('getting time long')
        self.D[self.time_col] = self.D[self.time_col].apply(pd.to_datetime)
        for i in self.time_col:
            self.D[i] = self.D[i].apply(lambda x:_calculate_time_long(x, base_date))

    def _refine_col_name(self):
        self.D = self.D.rename(columns = self.rename_pair)

    def _str_col_to_num(self):
        def _to_int_helper(x):
            try:
                b = int(x)
                return b
            except (TypeError, ValueError) as e:
                return -1

        self.D[self.str2num_col] = self.D[self.str2num_col].applymap(lambda x:_to_int_helper(x))

    def _drop_col(self):
        print(f'dropping cols')
        ori_len = len(self.D.columns)
        all_drop_col = list(set(self.drop_col + self.dummy_col))
        all_drop_col = [_ for _ in all_drop_col if _ not in self.protect_col]
        self.D.drop(all_drop_col, axis = 1, inplace = True)
        print(f'dropped {ori_len - len(self.D.columns)} cols')

    def _df_final_refine(self):
        for i in self.D.columns:
            self.D[i] = pd.to_numeric(self.D[i], errors = 'coerce')
        self.D = self.D.fillna(0)
        self.D = self.D.sort_values(by = self.id_col)
