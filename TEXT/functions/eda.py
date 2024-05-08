import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import plotly.express as px
import pandas as pd
import re

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

def lower_and_contractions(text, contractions):
    """
    Preprocesses a text.

    Args:
    -------
    text: str
        Text to preprocess.
    contractions: pandas.DataFrame
        DataFrame containing contractions.
    """
    # Normalize all commas to ' format and make all text lower case
    text = text.replace("’", "'").lower()
    # Create a dictionary to match contraction-expanded
    contractions_dict = dict(zip(contractions['contraction'], contractions['expanded']))
    # Expand the contractions
    words = text.split()
    expanded_words = [contractions_dict.get(word, word) for word in words]
    text = " ".join(expanded_words)
    return text

def stopwords_and_lemmatizer(text, stop_words, lemmatizer):
    """
    Preprocesses a text.

    Args:
    -------
    text: str
        Text to preprocess.
    stopwords: list
        List of stopwords.
    lemmatizer: nltk.stem.WordNetLemmatizer
        Lemmatizer.
    language: str, optional
        Language of the text. Defaults to "english".
    
    Returns:
    -------
    text: str
        The preprocessed text.
    """
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove digits
    text = re.sub(r'\d', '', text)
    # Remove stop words
    text = " ".join([word for word in text.split() if word not in stop_words])
    # Lemmatize words
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def get_word_counts(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Gets the number of occurrences of every word in a 
    pandas DataFrame column containing text.

    Args:
    -------
    data: pd.DataFrame
        Dataframe to process.
    col: str
        Name of the column to process.
    
    Returns:
    -------
    word_count: pd.DataFrame
        Dataframe with the number of occurrences of every word.
    """
    # Get the number of occurrences of every word in the dataset
    word_count = pd.Series(" ".join(data[col]).split()).value_counts()
    # Convert into a DataFrame
    word_count = pd.DataFrame(word_count).reset_index()
    # Rename columns
    word_count.columns = ["word", "frequency"]
    return word_count

def plot_top_n_words_frequency(word_counts: pd.DataFrame, word_col:str, freq_col:str, top_n:int, dataset:str, palette = "YlOrRd_r") -> None:
    """
    Plots the frequency distribution of the n most common words in the dataset.

    Args:
    -------
    word_counts: pd.DataFrame
        Dataframe with the number of occurrences of every word.
    word_col: str
        Name of the word column.
    freq_col: str
        Name of the frequency column.
    top_n: int
        Number of most common words to plot.
    dataset: str
        Name of the dataset
    palette: str
        Color palette to use.
    
    Returns:
    -------
    None
        The function plots the frequency distribution of the n most common words in the dataset.
    """
    # Plot the frequency distribution of the n most common words in the dataset
    plt.figure(figsize=(10,5), dpi=120)
    sns.barplot(data=word_counts.head(top_n), x=word_col, y=freq_col, palette=palette)
    plt.title(f'Frequency distribution of the {top_n} most common words in {dataset} dataset', fontsize=18)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=90)
    # Despine the plot
    sns.despine(left=True, bottom=True)
    plt.show()
    return

def plot_wordcloud(data: pd.DataFrame(), col: str, dataset:str) -> None:
    """
    Plots a wordcloud from a pandas DataFrame column.

    Args:
    -------
    data: pd.DataFrame
        Dataframe to plot.
    col: str
        Name of the column to plot.
    dataset: str
        Name of the dataset
    
    Returns:
    -------
    None
        The function plots a wordcloud.
    """
    # Join all the text together
    all_text = " ".join([i for i in data[col]])
    # Set the wordcloud parameters
    wc = WordCloud(
        background_color="white",
        width=1600,
        height=900,
        max_words=300, 
        contour_width=3, 
        contour_color='steelblue'
    )
    # Generate the wordcloud from the text
    wc.generate(all_text)
    # Plot the wordcloud
    print(f'{dataset} DATASET')
    plt.figure(figsize=(15, 8), facecolor='k')
    plt.imshow(wc)
    plt.axis("off")
    plt.show()
    return

def min_word_counts(df: pd.DataFrame):
    """
    Calculate word counts auxiliary DataFrame.

    Args:
    - data (pd.DataFrame): DataFrame with word frequencies.

    Returns:
    - pd.DataFrame: DataFrame with frequencies, number of words, and percentage.
    """
    frequency_values = [10, 5, 4, 3, 2]
    min_word_counts = pd.DataFrame({"frequency": frequency_values})
    min_word_counts["number_of_words"] = min_word_counts["frequency"].apply(lambda x: len(df[df["frequency"] <= x]))
    total_words = len(df)
    min_word_counts["percentage"] = (min_word_counts["number_of_words"] / total_words) * 100
    return min_word_counts

def filter_words_by_frequency(text, word_freq, threshold):
    """
    Given a text and a dictionary with the frequency of each word,
    return a text with the words with frequency less than 3 removed.

    Parameters
    ----------
    text : str
        Text to be filtered.
    word_freq : dict
        Dictionary with the frequency of each word.
    
    Returns
    -------
    str
        Text with the words with frequency less than 3 removed.
    """
    words = text.split()
    filtered_words = [word for word in words if word_freq.get(word, 0) > threshold]
    return " ".join(filtered_words)

