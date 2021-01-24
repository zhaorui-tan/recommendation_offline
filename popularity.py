import pandas as pd
from collections import Counter


class Popularity:
    """
    similarity table
    -------------------------
    popular_item     | int
    -------------------------
    popular_rate     | float
    -------------------------
    """

    def __init__(self, order: pd.DataFrame, target_col: str, top_n: int):
        self.O = order
        self.L = len(self.O)
        self.T = target_col
        self.N = top_n

    def calculate_most_n_popularity(self) -> pd.DataFrame:
        popular_item = []
        popular_rate = []
        most_n = self.N
        if len(self.O[self.T].drop_duplicates()) >= self.N:
            most_n = len(self.O[self.T].drop_duplicates())
        for (t, c) in Counter(self.O[self.T].values).most_common(most_n):
            popular_item.append(t)
            popular_rate.append(c / self.L)
        popular_df = pd.DataFrame(data = { 'popular_item':popular_item, 'popular_rate':popular_rate })
        return popular_df
