'''
This code implements recommender system using database of 16559 books including its summary, title.. extracted from the November 2, 2012 dump of English-language Wikipedia
The idea used here is based on the Latent Dirchlet Allocation (LDA) approach   learning as a part of Udacity NLP course
Step 1: Preprocess data
Step 2 : Implement LDA approach
Step3:  Recommender system
'''

'''
Loading Gensim and nltk libraries
'''
import pandas as pd
import re
import gensim
from gensim import similarities
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
from nltk.corpus import stopwords
np.random.seed(400)
from gensim import corpora, models



# Step 1: Preprocessing step: Loading the Data and removing all columns except book title and summary
d = pd.read_csv("data/booksummaries.txt", sep="\t", names=['article ID', 'Freebase ID', 'title','author','publ date','genre','summary'])
d=d.drop(['article ID','publ date', 'genre', 'author','Freebase ID'],axis=1)
# Convert to list
df = d.values.tolist()
df[1] = re.sub(r"[^a-zA-Z0-9]", " ",  str(df[1]).lower())
print("We have {} books extracted from the November 2, 2012 dump of English-language Wikipedia".format(len(d)))
dler = nltk.downloader.Downloader()
dler._update_index()
dler.download('wordnet')
stemmer = SnowballStemmer("english")

'''
Write a function to perform the pre processing steps on the entire dataset
'''
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
# Tokenize and lemmatize
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result


# Preprocess the summary of whole books: tokenization, lemmatize, stemming

processed_docs = d['summary'].map(preprocess)
print("Lets see how the data looks after preprocessing",processed_docs[:5])

# Collecting Bag of wordw
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above =0.1,keep_n=100000)
bow_corpus = [dictionary.doc2bow(document) for document in  processed_docs]


""" Uncomment the below lines to optimize the topics from the database"""
"""
import random
import re
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
topicnums = [1,5,10,15,20,25,30,35,40,45,50]
project_folder = os.getcwd()


ldamodels_bow = {}
for i in topicnums:
    random.seed(42)
    if not os.path.exists(project_folder+'/models/ldamodels_bow_'+str(i)+'.lda'):
        ldamodels_bow[i] = models.LdaModel(bow_corpus, num_topics=i, random_state=42, update_every=1, passes=10, id2word=dictionary)
        ldamodels_bow[i].save(project_folder+'/models/ldamodels_bow_'+str(i)+'.lda')
        print('ldamodels_bow_{}.lda created.'.format(i))
    else:
        print('ldamodels_bow_{}.lda already exists.'.format(i))
lda_topics = {}
for i in topicnums:
    lda_model = models.LdaModel.load(project_folder+'/models/ldamodels_bow_'+str(i)+'.lda')
    lda_topics_string = lda_model.show_topics(i)
    lda_topics[i] = ["".join([c if c.isalpha() else " " for c in topic[1]]).split() for topic in lda_topics_string]

pickle.dump(lda_topics,open(project_folder+'/models/pub_lda_bow_topics.pkl','wb'))
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return float(len(intersection))/float(len(union))


lda_stability = {}
for i in range(0, len(topicnums) - 1):
    jacc_sims = []
    for t1, topic1 in enumerate(lda_topics[topicnums[i]]):
        sims = []
        for t2, topic2 in enumerate(lda_topics[topicnums[i + 1]]):
            sims.append(jaccard_similarity(topic1, topic2))
        jacc_sims.append(sims)
    lda_stability[topicnums[i]] = jacc_sims

pickle.dump(lda_stability, open(project_folder + '/models/pub_lda_bow_stability.pkl', 'wb'))


lda_stability = pickle.load(open(project_folder+'/models/pub_lda_bow_stability.pkl','rb'))
mean_stability = [np.array(lda_stability[i]).mean() for i in topicnums[:-1]]
 with sns.axes_style("darkgrid"):
 #   x = topicnums[:-1]
  #  y = mean_stability
   # plt.figure(figsize=(20,10))
   # plt.plot(x,y,label='Average Overlap Between Topics')
    #plt.xlim([1, 55])
    plt.ylim([0, 0.25])
    plt.xlabel('Number of topics')
    plt.ylabel('Average Jaccard similarity')
    plt.title('Average Jaccard Similarity Between Topics')
    plt.legend()
    plt.show()
#import ipdb
#ipdb.set_trace()
"""
#
# Step 2: Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'

lda_model = gensim.models.LdaMulticore(bow_corpus,
                                      num_topics=10,
                                      id2word=dictionary,
                                      passes=30)

print("Let us see the set of topics obtained. Each topic is a collection of keywords, again, in a certain proportion.")
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(topic, idx ))
    print("\n")

""" Uncomment the below lines to visualize Word Clouds of Top N Keywords in Each Topic"""
""" 
# Visualize the topics
# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=gensim.parsing.preprocessing.STOPWORDS,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
"""

""" Step 3: Recommender system: Idea is to find the similarity matrix based on lda model and for each book, the 10 best matches 
from the corpus are obtained based on similarity matrix.
"""
corpus_lda_model = lda_model[bow_corpus]


index = similarities.MatrixSimilarity(lda_model[bow_corpus])
def book_recommender(title):
    books_checked = 0
    for i in range(len(d)):
        recommendation_scores = []
        if d['title'] [i] == title:
            lda_vectors = corpus_lda_model[i]
            sims = index[lda_vectors]
            sims = list(enumerate(sims))
            for sim in sims:
                book_num = sim[0]


                recommendation_score = [book_num, sim[1]]
                recommendation_scores.append(recommendation_score)

            recommendation = sorted(recommendation_scores, key=lambda x: x[1], reverse=True)
            print("Your book's most prominent tokens are:")
            article_tokens = bow_corpus[i]
            sorted_tokens = sorted(article_tokens, key=lambda x: x[1], reverse=True)
            sorted_tokens_10 = sorted_tokens[:10]
            for i in range(len(sorted_tokens_10)):
                print("Word {} (\"{}\") appears {} time(s).".format(sorted_tokens_10[i][0],
                                                                    dictionary[sorted_tokens_10[i][0]],
                                                                    sorted_tokens_10[i][1]))
            print('-----')
            print("Your book's most prominant topic is:")
            print(lda_model.print_topic(max(lda_vectors, key=lambda item: item[1])[0]))

            books_Ind =[i for i in recommendation[1:11]]

            books=[]
            for i in range(0, 10):
                books.append(d['title'][books_Ind[i][0]])
            print('-----')
            print('Here are your recommendations for "{}":'.format(title))
            print(books)


        else:
            books_checked += 1

        if books_checked == len(d):
            book_suggestions = []
            print('Sorry, but it looks like "{}" is not available.'.format(title))


""" Lets test the code now with A Clockwork Orange. This is the second book available in the Corpus"""
book_recommender("A Clockwork Orange")


