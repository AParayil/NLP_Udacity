# BeautifulSoup to easily remove HTML tags
from bs4 import BeautifulSoup
# RegEx for removing non-letter characters
import re
# NLTK library for the remaining steps
import nltk
nltk.download("stopwords")  # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords  # import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
from nltk.tokenize import word_tokenize

def review_to_words(review):
    text = BeautifulSoup(review, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    words = [PorterStemmer().stem(w) for w in words]
    return words


