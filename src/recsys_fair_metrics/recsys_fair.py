import pandas as pd
from .util.util import parallel_literal_eval
from .exposure.exposure_metrics import ExposureMetric
from typing import List, Dict, Any


class RecsysFair(object):
    def __init__(
        self,
        df: pd.DataFrame,
        user_metadata: pd.DataFrame = None,
        supp_metadata: pd.DataFrame = None,
        user_column: str = "user_id",
        item_column: str = "item_id",
        reclist_column: str = "reclist_column",
    ):
        self._user_column = user_column
        self._item_column = item_column
        self._reclist_column = reclist_column
        self._supp_metadata = supp_metadata
        self._user_metadata = user_metadata
        self._dataframe = df.copy()
        self.fit()

    def fit(self):
        self._dataframe[self._reclist_column] = parallel_literal_eval(
            self._dataframe[self._reclist_column]
        )
        self._dataframe["first_rec"] = self._dataframe[self._reclist_column].apply(
            lambda l: l[0]
        )

    def exposure(self, column: str, k: int = 10):
        return ExposureMetric(
            dataframe=self._dataframe,
            supp_metadata=self._supp_metadata,
            column=column,
            user_column=self._user_column,
            item_column=self._item_column,
            rec_list=self._reclist_column,
            k=k,
        )
