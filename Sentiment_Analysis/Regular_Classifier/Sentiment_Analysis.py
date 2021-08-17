## This code implements sentiment analysis using the popular imdb dataset based on the exercise in the Udacity NLP course
## Step 1: reviews are provided in the folder data, therefore first step is to convert the data into a dictionary containing
# positive and negative reviews for both test and training set
## Step 2 : Preprocess the data to remove html tags, convert to lower case, tokenize, stemming, etc.
## Step3: Bag of words to extract features and normalize it
## This approach applies both Naive Bayes and Decision Tree for Classification

# Required packages
import os
import sklearn.preprocessing as pr
import pickle
import random
from sklearn.naive_bayes import  GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
################################################################################################

# Set the vocabulary size
from SentimentAnalysis.read_imdb import read_imdb_data as read_data
from SentimentAnalysis.read_imdb import prepare_imdb_data
from SentimentAnalysis.visualize_wordcloud import visualize
from SentimentAnalysis.preprocessing_data import  review_to_words
from SentimentAnalysis.bag_of_words import  extract_BoW_features
################################################################################################

### Step 1:
data, labels = read_data()
print("IMDb reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
        len(data['train']['pos']), len(data['train']['neg']),
        len(data['test']['pos']), len(data['test']['neg'])))

# To visualize reviews using wordcloud
visualize(data,show_plt=1)

# combine pos  and neg. reviews to get a test and training set
data_train, data_test, labels_train, labels_test = prepare_imdb_data(data,labels)
print("IMDb reviews (combined): train = {}, test = {}".format(len(data_train), len(data_test)))





cache_dir = os.path.join("cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists


def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
        """Convert each review to words; read from cache if available."""

        # If cache_file is not None, try to read from it first
        cache_data = None
        if cache_file is not None:
                try:
                        with open(os.path.join(cache_dir, cache_file), "rb") as f:
                                cache_data = pickle.load(f)
                        print("Read preprocessed data from cache file:", cache_file)
                except:
                        pass  # unable to read from cache, but that's okay

        # If cache is missing, then do the heavy lifting
        if cache_data is None:
                # Preprocess training and test data to obtain words for each review
                words_train = list(map(review_to_words, data_train))
                words_test = list(map(review_to_words, data_test))

                # Write to cache file for future runs
                if cache_file is not None:
                        cache_data = dict(words_train=words_train, words_test=words_test,
                                          labels_train=labels_train, labels_test=labels_test)
                        with open(os.path.join(cache_dir, cache_file), "wb") as f:
                                pickle.dump(cache_data, f)
                        print("Wrote preprocessed data to cache file:", cache_file)
        else:
                # Unpack data loaded from cache file
                words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                                                                      cache_data['words_test'],
                                                                      cache_data['labels_train'],
                                                                      cache_data['labels_test'])

        return words_train, words_test, labels_train, labels_test


# Step 2: Preprocess data
words_train, words_test, labels_train, labels_test = preprocess_data(
        data_train, data_test, labels_train, labels_test)

# Take a look at a sample
print("\n--- Raw review ---")
print(data_train[1])
print("\n--- Preprocessed words ---")
print(words_train[1])
print("\n--- Label ---")
print(labels_train[1])


# Step3: Extract Bag of Words features for both training and test datasets
features_train, features_test, vocabulary = extract_BoW_features(words_train, words_test,cache_dir)
# Inspect the vocabulary that was computed
print("Vocabulary: {} words".format(len(vocabulary)))


print("Sample words: {}".format(random.sample(list(vocabulary.keys()), 8)))

# Sample
print("\n--- Preprocessed words ---")
print(words_train[5])
print("\n--- Bag-of-Words features ---")
print(features_train[5])
print("\n--- Label ---")
print(labels_train[5])

# Normalize Features in bag of words
features_train = pr.normalize(features_train, axis=1)
features_test = pr.normalize(features_test,axis =1)

# Step 4: Classification using Gaussian Naive Bayes
classifier = GaussianNB()
classifier.fit(features_train, labels_train)
print("{} Accuracy: train {}:,test{}:".format(classifier.__class__.__name__,\
                                             classifier.score(features_train,labels_train),classifier.score(features_test,labels_test)))

# classification using Gradient Boost Decision Tree Classifier
n_estimators=32

def GradientBoosting(x_train, y_train,x_test,y_test):
        clf1 = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=0.1,max_depth=1, random_state=0)
        clf1.fit(x_train,y_train)
        print("{} Accuracy: Training set {}:, Test set {}:".format(clf1.__class__.__name__,\
                                                                    clf1.score(x_train,y_train),\
                                                                    clf1.score(x_test,y_test)))
        return clf1

clf1 = GradientBoosting(features_train,labels_train,features_test,labels_test)