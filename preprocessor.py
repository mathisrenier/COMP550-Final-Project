# COMP 550 FINAL PROJECT

import loader
import nltk
import sys
import string
import pandas as pd
import numpy as np
import weka.core.jvm as jvm
import weka.core.packages as packages
import weka.core.dataset as dataset
import weka.core.converters as converters
from weka.filters import Filter


def install_package(pkg):
    # install package if necessary
    if not packages.is_installed(pkg):
        print("Installing %s..." % pkg)
        packages.install_package(pkg)
        print("Installed %s, please re-run script!" % pkg)
        jvm.stop()
        sys.exit(0)
    print('Package already installed.')


def remove_package(pkg):
    if packages.is_installed(pkg):
        print("Removing %s..." % pkg)
        packages.uninstall_package(pkg)
        print("Removed %s, please re-run script!" % pkg)
        jvm.stop()
        sys.exit(0)
    print('No such package is installed')


def preprocessor(instances, stopwords=False, stemming=False):
    sw = nltk.corpus.stopwords.words('english') + list(string.punctuation) if stopwords else []

    tweets = []
    sentiments = []
    intensities = []

    for k, v in instances.items():
        tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, reduce_len=True)
        tokens = tokenizer.tokenize(v.tweet)

        if stemming:
            stemmer = nltk.stem.PorterStemmer()
            processed_tokens = [stemmer.stem(w) for w in tokens if w not in sw]
        else:
            lemmatizer = nltk.stem.WordNetLemmatizer()
            processed_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in sw]

        v.tweet = ' '.join(w for w in processed_tokens)
        v.tweet = v.tweet.replace(',', '')  # to avoid errors if we generate a csv

    return instances


def extract_vectors(file):
    data_frame = pd.read_csv(file, skiprows=1, header=None, usecols=range(1,44))
    vectors = data_frame.to_numpy()
    return vectors


def affective_vectorizer(tweets_train, tweets_dev, tweets_test):
    jvm.start(packages=True)
    install_package('AffectiveTweets')

    data_train = dataset.create_instances_from_lists([[t] for t in tweets_train])
    data_dev = dataset.create_instances_from_lists([[t] for t in tweets_dev])
    data_test = dataset.create_instances_from_lists([[t] for t in tweets_test])

    filter = Filter(classname='weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector',
                    options=['-F', '-D', '-R', '-A', '-T', '-L', '-N', '-P', '-J', '-H', '-Q',
                             '-stemmer', 'weka.core.stemmers.NullStemmer',
                             '-stopwords-handler', 'weka.core.tokenizers.TweetNLPTokenizer',
                             '-I', '1', '-U',
                             '-tokenizer', 'weka.core.tokenizers.TweetNLPTokenizer'])
    filter.inputformat(data_train)
    filtered_data = filter.filter([data_train, data_dev, data_test])

    converters.save_any_file(filtered_data[0], 'data/affect-vectors/train.csv')
    converters.save_any_file(filtered_data[1], 'data/affect-vectors/dev.csv')
    converters.save_any_file(filtered_data[2], 'data/affect-vectors/test.csv')

    jvm.stop()


def get_ordered_lists(instances):
    tweets = []
    sentiments = []
    intensities = []

    for k, v in instances.items():
        tweets.append(v.tweet)
        sentiments.append(v.sentiment)
        intensities.append(v.intensity)

    return tweets, sentiments, intensities


if __name__ == '__main__':
    train_instances, dev_instances, test_instances = loader.load_instances()

    train_instances = preprocessor(train_instances, stopwords=True)
    dev_instances = preprocessor(dev_instances, stopwords=True)
    test_instances = preprocessor(test_instances, stopwords=True)

    tweets_train, sentiments_train, intensities_train = get_ordered_lists(train_instances)
    tweets_dev, sentiments_dev, intensities_dev = get_ordered_lists(dev_instances)
    tweets_test, sentiments_test, intensities_test = get_ordered_lists(test_instances)

    # affective_vectorizer(tweets_train, tweets_dev, tweets_test)

    affect_vect_train = extract_vectors('data/affect-vectors/train.csv')
    affect_vect_dev = extract_vectors('data/affect-vectors/dev.csv')
    affect_vect_test = extract_vectors('data/affect-vectors/test.csv')
