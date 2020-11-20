import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, TweetTokenizer
np.random.seed(seed=234)
from io import StringIO

# The below reads in N lines of text from the 40-million-word news corpus I used (provided by SwiftKey for educational purposes).
N = 10000
with open("ptb.train.txt") as myfile:
    articles = [next(myfile) for x in range(N)]
joined_articles = [" ".join(articles)]

# The below takes out anything that's not a letter, replacing it with a space, as well as any single letter that is not the pronoun "I" or the article "a."
def clean_article(article):
    art1 = re.sub("[^A-Za-z]", ' ', article)
    art2 = re.sub("\s[B-HJ-Zb-hj-z]\s", ' ', art1)
    art3 = re.sub("^[B-HJ-Zb-hj-z]\s", ' ', art2)
    art4 = re.sub("\s[B-HJ-Zb-hj-z]$", ' ', art3)
    return art4.lower()

# The below breaks up the words into n-grams of length 1 to 5 and puts their counts into a Pandas dataframe with the n-grams as column names.  The maximum number of n-grams can be specified if a large corpus is being used.
ngram_bow = CountVectorizer(stop_words = None, preprocessor = clean_article, tokenizer = WhitespaceTokenizer().tokenize, ngram_range=(1,5), max_features = None, max_df = 1.0, min_df = 1, binary = False)
ngram_count_sparse = ngram_bow.fit_transform(joined_articles)
ngram_count = pd.DataFrame(ngram_count_sparse.toarray())
ngram_count.columns = ngram_bow.get_feature_names()

# The below turns the n-gram-count dataframe into a Pandas series with the n-grams as indices for ease of working with the counts.  The second line can be used to limit the n-grams used to those with a count over a cutoff value.
sums = ngram_count.sum(axis = 0)
sums = sums[sums > 0]
ngrams = list(sums.index.values)

# The function below gives the total number of occurrences of 1-grams in order to calculate 1-gram frequencies
def number_of_onegrams(sums):
    onegrams = 0
    for ng in ngrams:
        ng_split = ng.split(" ")
        if len(ng_split) == 1:
            onegrams += sums[ng]
    return onegrams


# The function below makes a series of 1-gram frequencies.  This is the last resort of the back-off algorithm if the n-gram completion does not occur in the corpus with any of the prefix words.
def base_freq(og):
    freqs = pd.Series()
    for ng in ngrams:
        ng_split = ng.split(" ")
        if len(ng_split) == 1:
            freqs[ng] = sums[ng] / og
    return freqs


# For use in later functions so as not to re-calculate multiple times:
bf = base_freq(number_of_onegrams(sums))

# The function below finds any n-grams that are completions of a given prefix phrase with a specified number (could be zero) of words 'chopped' off the beginning.  For each, it calculates the count ratio of the completion to the (chopped) prefix, tabulating them in a series to be returned by the function.  If the number of chops equals the number of words in the prefix (i.e. all prefix words are chopped), the 1-gram base frequencies are returned.
def find_completion_scores(prefix, chops, factor = 0.4):
    cs = pd.Series()
    prefix_split = prefix.split(" ")
    l = len(prefix_split)
    prefix_split_chopped = prefix_split[chops:l]
    new_l = l - chops
    if new_l == 0:
        return factor**chops * bf
    prefix_chopped = ' '.join(prefix_split_chopped)
    for ng in ngrams:
        ng_split = ng.split(" ")
        if (len(ng_split) == new_l + 1) and (ng_split[0:new_l] == prefix_split_chopped):
            cs[ng_split[-1]] = factor**chops * sums[ng] / sums[prefix_chopped]
    return cs


# The below tries different numbers of 'chops' up to the length of the prefix to come up with a (still unordered) combined list of scores for potential completions of the prefix.
def score_given(given, fact = 0.4):
    sg = pd.Series()
    given_split = given.split(" ")
    given_length = len(given_split)
    for i in range(given_length+1):
        fcs = find_completion_scores(given, i, fact)
        for i in fcs.index:
            if i not in sg.index:
                sg[i] = fcs[i]
    return sg

#The below takes the potential completion scores, puts them in descending order and re-normalizes them as a percentage (pseudo-probability).
def score_output(given, fact = 0.4):
    sg = score_given(given, fact)
    ss = sg.sum()
    sg = 100 * sg / ss
    sg.sort_values(axis=0, ascending=False, inplace=True)
    sg = round(sg,1)[0:15]
    return sg.to_json()

