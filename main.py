from data_processor import DataProcessing
from CF import CF
from popularity import Popularity
from similarity import Similarity
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('../user_data.csv')
    i = None
    order_data = pd.read_csv('../order_data.csv')

    # processing user data
    drop_col = ['grp_cust_id', 'grp_name', 'contact_phn', 'owner_name', 'vpmn_id', 'rec_state', 'exp_date',
                'grp_cust_id', 'create_date', 'grp_ind_type_3', 'concat_grp_type']
    dummy_col = ['grp_state', 'grp_ind_type', 'sign_type', 'trade_grp_clust', 'grp_ind_type_1', 'grp_ind_type_2', ]
    time_col = ['eff_date']
    str2num_col = ['grp_county_id', 'grp_city_id', ]
    rename_pair = { 'grp_id':'group_id' }
    user_group_type = data[['grp_id', 'concat_grp_type']].rename(columns = rename_pair)
    up = DataProcessing(data, drop_col, dummy_col, time_col, rename_pair, str2num_col, 'group_id')
    user_data = up.D

    # Similarity
    # group by concat_grp_type
    common_pop_n = Popularity(order_data, 'plan_id', 10).calculate_most_n_popularity()
    pop_n_by_type = pd.DataFrame()
    for group, data in order_data.groupby('concat_grp_type'):
        pop_n = Popularity(data, 'plan_id', 10).calculate_most_n_popularity()
        if len(pop_n) < 10:
            pop_n = pop_n.append(common_pop_n[:10 - len(pop_n)], ignore_index = True)
        pop_n['concat_grp_type'] = [group] * len(pop_n)
        pop_n_by_type = pop_n_by_type.append(pop_n, ignore_index = True)

    ICsim = Similarity(mode = 'IC', order = order_data, target_col = 'plan_id', content_col = 'group_id',
                       top_n = 5).get_similarity_df()
    UCsim = Similarity(mode = 'UC', order = order_data, target_col = 'group_id', content_col = 'plan_id',
                       top_n = 5).get_similarity_df()

    # recommend by county
    for group, data in user_data.groupby('grp_city_id'):
        ULsim = Similarity(mode = 'UL', user_info = data, target_col = 'group_id', content_col = 'group_id',
                           top_n = 5).get_similarity_df()
        Usim = pd.concat([UCsim, ULsim], axis = 0).sort_values(['target_id', 'similarity'], ascending = False)
        cf_for_user = CF(Usim, ICsim, 10, order_data, 'group_id', 'plan_id', pop_n_by_type,
                         user_group_type = user_group_type).get_recommend_contents_df()
        cf_for_product = CF(ICsim, Usim, 10, order_data, 'plan_id', 'group_id').get_recommend_contents_df()
        cf_for_user['type'] = ['user'] * len(cf_for_user)
        cf_for_product['type'] = ['product'] * len(cf_for_product)
        all_recommends = pd.concat((cf_for_product, cf_for_user), ignore_index = True)
        rename = { 0:'r_0', 1:'r_1', 2:'r_2', 3:'r_3', 4:'r_4', 5:'r_5', 6:'r_6', 7:'r_7',
                   8:'r_8', 9:'r_9' }
        all_recommends.rename(columns = rename, inplace = True)
        print(all_recommends)
