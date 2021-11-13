import pandas as pd
import numpy as np
from typing import List, Dict, Any
import functools

import os
from tqdm import tqdm
from multiprocessing.pool import Pool
import functools

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from ..util.util import mean_confidence_interval
from ..util.rank_metrics import ndcg_at_k
from ..util.rank_metrics import ndcg_at_k, precision_at_k

TEMPLATE = "plotly_white"


def _create_relevance_list(
    sorted_actions: List[Any], expected_action: Any
) -> List[int]:
    return [
        1 if str(action) == str(expected_action) else 0 for action in sorted_actions
    ]


# def _color_by_metric(metric):
#     if "ndcg" in metric:
#         return "#C44E51"
#     elif "coverage" in metric:
#         return "#DD8452"
#     elif "personalization" in metric:
#         return "#55A868"
#     elif "count" in metric:
#         return "#CCB974"
#     else:
#         return "#8C8C8C"


class ExposureMetric(object):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        supp_metadata: pd.DataFrame,
        column: str,
        user_column: str,
        item_column: str,
        rec_list: str,
        k: int,
    ):
        self._dataframe = dataframe.fillna("-")
        self._supp_metadata = supp_metadata
        self._rec_list = rec_list
        self._user_column = user_column
        self._item_column = item_column
        self._column = column
        self._df_metrics = None
        self._df_reclist = None
        self._k = k
        self.fit(
            self._dataframe, self._column, self._user_column, self._rec_list, self._k
        )

    def fit(
        self,
        df: pd.DataFrame,
        column: str,
        user_column: str,
        rec_list: str,
        k: int,
    ) -> pd.DataFrame:

        df = df[[user_column, rec_list]].copy()
        df[rec_list] = df[rec_list].apply(lambda l: l[:k])
        df["pos"] = df[rec_list].apply(lambda l: list(range(len(l))))

        # Explode reclist
        df_sup_per_user = (
            df.set_index([user_column]).apply(pd.Series.explode).reset_index()
        )
        df_sup_per_user["count"] = 1
        # print(df_sup_per_user.shape)
        df_sup_exp = df_sup_per_user.groupby(rec_list).count().sort_values(user_column)

        # Prob Of recommender
        df_sup_prob = df_sup_exp[["count"]] / len(df)
        df_sup_prob = df_sup_prob.rename(columns={"count": "prob"})

        # Mean Score pivot per column
        df_sup_prob_pos = pd.pivot_table(
            df_sup_per_user,
            index=self._rec_list,
            columns="pos",
            values="count",
            aggfunc=np.sum,
        ).fillna(0.001)
        sum_values = np.array(df_sup_prob_pos.sum(axis=1).values)
        df_sup_prob_pos = df_sup_prob_pos / np.array([sum_values]).transpose()

        # Calcule exposure per group
        # ...

        self._df_cum_exposure = df_sup_exp
        self._df_sup_prob = df_sup_prob
        self._df_sup_prob_pos = df_sup_prob_pos
        self._df_reclist = df[[user_column, rec_list]]
        self.all_supp = list(np.unique(df_sup_per_user[rec_list].values))

    def metric(self, prop: int = 10000):
        """
        All exposure metrics
        """

        m = (
            self.df_exposure(prop)
            .groupby("column")
            .agg(
                supp_size=("prob_exp", "count"),
                exp_cum=("prob_exp", "sum"),
                exp_mean=("prob_exp", "mean"),
            )
            .to_dict()
        )

        return m

    def prob_exp(self, supp: List, prop: int):
        w = np.log2(np.arange(2, self._k + 2))
        # w = np.ones(self._k)
        values = []
        for s in supp:
            rec_prob = self._df_sup_prob.loc[s].prob
            rec_prob_pos = self._df_sup_prob_pos.loc[s].values
            v = np.ma.masked_invalid((prop * (rec_prob * rec_prob_pos)) / w).sum()
            values.append(v)

        return np.sum(values)

    def ndce_at_k(self, supp: str):
        """
        Normalized Discounted Cumulative Exposure (NDCE)
        """
        df = self._df_reclist.copy()
        df["item_column"] = supp

        with Pool(os.cpu_count()) as p:
            print("Creating the relevance lists...")
            df["relevance_list"] = list(
                tqdm(
                    p.starmap(
                        _create_relevance_list,
                        zip(df[self._rec_list], df["item_column"]),
                    ),
                    total=len(df),
                )
            )

            print("Calculating ndce@k...")
            df["ndce@{}".format(self._k)] = list(
                tqdm(
                    p.map(
                        functools.partial(ndcg_at_k, k=self._k), df["relevance_list"]
                    ),
                    total=len(df),
                )
            )

        return df["ndce@{}".format(self._k)].mean()

    def df_exposure(self, prop: int):
        df = self._df_cum_exposure[["count"]].reset_index()
        df["prob_exp"] = df[self._rec_list].apply(
            lambda supp: self.prob_exp([supp], prop)
        )

        df_group = (
            self._supp_metadata[[self._item_column, self._column]]
            .drop_duplicates()
            .fillna("-")
        )
        df_group["column"] = df_group[self._column].apply(
            lambda c: self._column + "." + str(c)
        )

        exp_cum = df.merge(df_group, left_on=self._rec_list, right_on=self._item_column)

        return exp_cum

    def df_position_exposure(self):
        position_exposure = self._df_sup_prob_pos.reset_index()
        df_group = (
            self._supp_metadata[[self._item_column, self._column]]
            .drop_duplicates()
            .fillna("-")
        )
        position_exposure = position_exposure.merge(
            df_group, left_on=self._rec_list, right_on=self._item_column
        )
        position_exposure = (
            position_exposure.groupby(by=self._column)
            .mean()
            .drop(columns=[self._rec_list, self._item_column])
        )
        position_exposure.columns = position_exposure.columns + 1

        return position_exposure[position_exposure.columns[::-1]]

    def show(self, kind: str = "geral", **kwargs):
        if kind == "geral":
            return self.show_geral(**kwargs)
        elif kind == "per_group":
            return self.show_per_group(**kwargs)
        elif kind == "per_group_norm":
            return self.show_per_group_norm(**kwargs)
        elif kind == "per_rank_pos":
            return self.show_per_rank_pos(**kwargs)

    def show_per_group_norm(self, **kwargs):
        """ """
        prop = 1 if "prop" not in kwargs else kwargs["prop"]
        with_annotate = (
            True if "with_annotate" not in kwargs else kwargs["with_annotate"]
        )

        title = "Exposure per Group" if "title" not in kwargs else kwargs["title"]
        data = []
        exp_cum = self.df_exposure(prop)
        pos_exp = self.df_position_exposure()

        prob_exp = exp_cum.groupby(self._column).sum()["prob_exp"].values
        count = exp_cum.groupby(self._column).count()["column"].values
        norm_factor = 1 / sum(prob_exp / count)

        exp_final = []
        for group, rows in exp_cum.groupby(self._column):
            size_supp = [i + 1 for i in range(len(rows))]
            values = (rows["prob_exp"].cumsum() / size_supp) * norm_factor

            if len(rows) == 1:
                x = np.array(list(range(len(rows)))) * 100
            else:
                x = np.array(list(range(len(rows)))) * 100 / (len(rows) - 1)
            data.append(
                go.Scatter(
                    name=self._column + "." + str(group),
                    x=x,  # np.array(list(range(len(rows)))) * 100 / (len(rows) - 1),
                    y=values,
                )
            )

            exp_final.append(values.max())

        fig = go.Figure(data=data)

        # Change the bar mode
        fig.update_layout(
            template=TEMPLATE,
            legend_orientation="h",
            xaxis_title="% Producers",
            yaxis_tickformat=".1%",
            # yaxis=dict(ticksuffix="%"),
            yaxis_title="% Exposure".format(prop * self._k),
            legend=dict(y=-0.2),
            title=title,
        )

        if with_annotate:
            for a in exp_final:
                fig.add_annotation(
                    x=100,
                    y=a,
                    xref="x",
                    yref="y",
                    text="{}".format(np.round(a, 2)),
                    showarrow=True,
                    font=dict(
                        family="Courier New, monospace", size=12, color="#ffffff"
                    ),
                    align="center",
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="#636363",
                    ax=30,
                    ay=-10,
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=2,
                    bgcolor="#ff7f0e",
                    opacity=0.7,
                )

        return fig

    def show_per_rank_pos(self, **kwargs):
        """ """
        title = (
            "Exposure per Rank Position" if "title" not in kwargs else kwargs["title"]
        )

        pos_exp = self.df_position_exposure()

        fig = px.bar(pos_exp, color_discrete_sequence=px.colors.sequential.solar)
        fig.update_layout(
            legend_title_text="Rank",
            template=TEMPLATE,
            yaxis_tickformat=".1%",
            #  yaxis=dict(ticksuffix="%"),
            legend={"traceorder": "reversed"},
            title=title,
        )
        fig.update_yaxes(title="")

        return fig

    def show_per_group(self, **kwargs):
        """ """
        prop = 10000 if "prop" not in kwargs else kwargs["prop"]
        with_annotate = (
            True if "with_annotate" not in kwargs else kwargs["with_annotate"]
        )

        title = "Exposure per Group" if "title" not in kwargs else kwargs["title"]
        data = []

        exp_cum = self.df_exposure(prop)

        exp_final = []
        for group, rows in exp_cum.groupby(self._column):
            size_supp = [i + 1 for i in range(len(rows))]
            values = rows["prob_exp"].cumsum() / size_supp

            data.append(
                go.Scatter(
                    name=self._column + "." + str(group),
                    x=np.array(list(range(len(rows)))) * 100 / (len(rows) - 1),
                    y=values,
                )
            )

            exp_final.append(values.max())

        fig = go.Figure(data=data)

        # Change the bar mode
        fig.update_layout(
            template=TEMPLATE,
            legend_orientation="h",
            xaxis_title="% Producers",
            yaxis_title="Exposure 1:{}".format(prop * self._k),
            legend=dict(y=-0.2),
            title=title,
        )

        if with_annotate:
            for a in exp_final:
                fig.add_annotation(
                    x=100,
                    y=a,
                    xref="x",
                    yref="y",
                    text="{}".format(np.round(a, 2)),
                    showarrow=True,
                    font=dict(
                        family="Courier New, monospace", size=12, color="#ffffff"
                    ),
                    align="center",
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="#636363",
                    ax=30,
                    ay=-10,
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=2,
                    bgcolor="#ff7f0e",
                    opacity=0.7,
                )

        return fig

    def show_geral(self, **kwargs):
        """ """

        # top_k  = 10 if 'top_k' not in kwargs else kwargs['top_k']
        title = "Exposure" if "title" not in kwargs else kwargs["title"]

        df = self._df_cum_exposure
        data = []

        exp_cum = list(df[self._user_column].cumsum()) / df[self._user_column].sum()

        data.append(
            go.Scatter(
                name="Reclist",
                x=np.array(list(range(len(exp_cum)))) * 100 / len(exp_cum),
                y=exp_cum * 100,
            )
        )

        data.append(
            go.Scatter(
                name="Equality",
                x=list(range(100)),
                y=list(range(100)),
                mode="markers",
                marker=dict(color="rgb(128, 128, 128)", size=2, opacity=0.7),
            )
        )

        fig = go.Figure(data=data)

        # Change the bar mode
        fig.update_layout(
            template=TEMPLATE,
            legend_orientation="h",
            xaxis_title="% Producers",
            yaxis_title="% Exposure",
            legend=dict(y=-0.2),
            title=title,
        )

        # fig.add_annotation(
        #     x=0.8,
        #     y=0.1,
        #     xref="x",
        #     yref="y",
        #     text="Max K-S: {}".format(self.metric().round(3)),
        #     showarrow=False,
        #     font=dict(
        #         family="Courier New, monospace",
        #         size=12,
        #         color="#ffffff"
        #         ),
        #     align="center",
        #     bordercolor="#c7c7c7",
        #     borderwidth=1,
        #     borderpad=4,
        #     bgcolor="#ff7f0e",
        #     opacity=0.8
        # )

        return fig
