from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    return px.line(df, x=x, y=y, title=title)


def multi_line_chart(df: pd.DataFrame, x: str, ys: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for y in ys:
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=y))
    fig.update_layout(title=title)
    return fig


def scatter_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    return px.scatter(df, x=x, y=y, title=title, trendline="ols")


def heatmap_corr(df: pd.DataFrame, cols: list[str], title: str) -> go.Figure:
    corr = df[cols].corr()
    return px.imshow(corr, text_auto=True, title=title)


def drawdown_chart(df: pd.DataFrame, equity_col: str, title: str) -> go.Figure:
    d = df.copy()
    d["peak"] = d[equity_col].cummax()
    d["drawdown"] = d[equity_col] / d["peak"] - 1.0
    return px.line(d, x="date", y="drawdown", title=title)
