import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import plotly.express as px
import pandas as pd

def check_class_imbalance(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Checks the class imbalance in a pandas DataFrame column.

    Args:
    -------
    data: pd.DataFrame
        Dataframe to process.
    col: str
        Name of the column to process.
    
    Returns:
    -------
    pd.DataFrame:
        Dataframe with the absolute and relative frequencies of each class.
    """
    return (data
        .groupby(by=col)
        # Get the absolute frequencies
        .agg(Freq=(col, "count"))
        # Compute the relative frequencies
        .assign(RelFreq = lambda x: x["Freq"] / x["Freq"].sum())
        # Reset the index and sort by label
        .reset_index()
        .sort_values(by=col)
    )

