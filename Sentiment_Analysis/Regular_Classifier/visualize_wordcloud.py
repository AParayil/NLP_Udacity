import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
sentiment='pos'
def visualize(data,show_plt):
    combined_text = " ".join([review for review in data['train'][sentiment]])
    # next step is to visualize the reviews
    wc = WordCloud(stopwords=STOPWORDS.update(['br', 'film', 'movie']), background_color='white', min_word_length=2,
                   max_words=50)
    if show_plt:
        plt.imshow(wc.generate(combined_text))
        plt.show()


# Now we need to join the list elements without a separator
