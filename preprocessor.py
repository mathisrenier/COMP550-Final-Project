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
    # install weka package if necessary
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
    '''
    Preprocesses each tweet in instances.
    :param instances: Dict of EmoIntInstance
    :param stopwords: If True, removes stopwords
    :param stemming:  Stems if True, other wize lemmatizes
    :return: Preprocessed dict of EmoIntInstance
    '''
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
    '''
    Creates a 2d numpy array from a csv file
    :param file: Path of the file (str)
    :return: A 2d numpy array of the tweet vectors
    '''
    data_frame = pd.read_csv(file, skiprows=1, header=None, usecols=range(1,44))
    vectors = data_frame.to_numpy()
    return vectors


def affective_vectorizer(tweets, filename):
    '''
    Vectorizes the tweets and saves the vectors as csv.
    :param tweets: list of tweets
    :param filename: name of the saved file
    '''
    jvm.start(packages=True)
    install_package('AffectiveTweets')

    data = dataset.create_instances_from_lists([[t] for t in tweets])

    filter = Filter(classname='weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector',
                    options=['-F', '-D', '-R', '-A', '-T', '-L', '-N', '-P', '-J', '-H', '-Q',
                             '-stemmer', 'weka.core.stemmers.NullStemmer',
                             '-stopwords-handler', 'weka.core.tokenizers.TweetNLPTokenizer',
                             '-I', '1', '-U',
                             '-tokenizer', 'weka.core.tokenizers.TweetNLPTokenizer'])
    filter.inputformat(data)
    filtered_data = filter.filter(data)

    converters.save_any_file(filtered_data, 'data/affect-vectors/'+filename)

    jvm.stop()


def get_ordered_lists(instances):
    '''
    Creates lists from the dict of instances
    :param instances: dict of EmoIntInstance
    :return: 3 lists of the elements of instances, in the same order
    '''
    tweets = []
    sentiments = []
    intensities = []

    for k, v in instances.items():
        tweets.append(v.tweet)
        sentiments.append(v.sentiment)
        intensities.append(v.intensity)

    return tweets, sentiments, intensities


def get_affect_vectors(type='train', emotion='anger'):
    '''
    :param type: 'train' or 'test'
    :param emotion: 'anger', 'fear', 'joy' or 'sad'
    :return: A tuple (vectors, intensities), with vectors being a 2d numpy array and intensities being a list of floats.
    '''
    assert type == 'train' or type == 'test', 'Invalid instance type, try train/test'
    assert emotion == 'anger' or emotion == 'fear' or emotion == 'joy' or emotion == 'sad', 'Invalid emotion, try anger/fear/joy/sad'

    # load separated instances
    anger_train, fear_train, joy_train, sad_train = loader.load_instances_separated(type='train')
    anger_test, fear_test, joy_test, sad_test = loader.load_instances_separated(type='test')

    # preprocess separated instances
    for i in [anger_train, fear_train, joy_train, sad_train, anger_test, fear_test, joy_test, sad_test]:
        i = preprocessor(i, stopwords=True)

    # create ordered lists from separated instances
    tweets_anger_train, sentiments_anger_train, intensities_anger_train = get_ordered_lists(anger_train)
    tweets_fear_train, sentiments_fear_train, intensities_fear_train = get_ordered_lists(fear_train)
    tweets_joy_train, sentiments_joy_train, intensities_joy_train = get_ordered_lists(joy_train)
    tweets_sad_train, sentiments_sad_train, intensities_sad_train = get_ordered_lists(sad_train)

    tweets_anger_test, sentiments_anger_test, intensities_anger_test = get_ordered_lists(anger_test)
    tweets_fear_test, sentiments_fear_test, intensities_fear_test = get_ordered_lists(fear_test)
    tweets_joy_test, sentiments_joy_test, intensities_joy_test = get_ordered_lists(joy_test)
    tweets_sad_test, sentiments_sad_test, intensities_sad_test = get_ordered_lists(sad_test)

    # create csv vector files for separated tweets
    # affective_vectorizer(tweets_anger_train, 'train/anger.csv')
    # affective_vectorizer(tweets_fear_train, 'train/fear.csv')
    # affective_vectorizer(tweets_joy_train, 'train/joy.csv')
    # affective_vectorizer(tweets_sad_train, 'train/sad.csv')
    #
    # affective_vectorizer(tweets_anger_test, 'test/anger.csv')
    # affective_vectorizer(tweets_fear_test, 'test/fear.csv')
    # affective_vectorizer(tweets_joy_test, 'test/joy.csv')
    # affective_vectorizer(tweets_sad_test, 'test/sad.csv')

    # extract vectors from csv for separated datasets
    anger_vect_train = extract_vectors('data/affect-vectors/train/anger.csv')
    fear_vect_train = extract_vectors('data/affect-vectors/train/fear.csv')
    joy_vect_train = extract_vectors('data/affect-vectors/train/joy.csv')
    sad_vect_train = extract_vectors('data/affect-vectors/train/sad.csv')

    anger_vect_test = extract_vectors('data/affect-vectors/test/anger.csv')
    fear_vect_test = extract_vectors('data/affect-vectors/test/fear.csv')
    joy_vect_test = extract_vectors('data/affect-vectors/test/joy.csv')
    sad_vect_test = extract_vectors('data/affect-vectors/test/sad.csv')

    return {
        'train': {
            'anger': (anger_vect_train, intensities_anger_train),
            'fear': (fear_vect_train, intensities_fear_train),
            'joy': (joy_vect_train, intensities_joy_train),
            'sad': (sad_vect_train, intensities_sad_train),
        },
        'test': {
            'anger': (anger_vect_test, intensities_anger_test),
            'fear': (fear_vect_test, intensities_fear_test),
            'joy': (joy_vect_test, intensities_joy_test),
            'sad': (sad_vect_test, intensities_sad_test),
        },
    }[type][emotion]


if __name__ == '__main__':
    # load whole instances
    train_instances, test_instances = loader.load_instances()

    # load separated instances
    anger_train, fear_train, joy_train, sad_train = loader.load_instances_separated(type='train')
    anger_test, fear_test, joy_test, sad_test = loader.load_instances_separated(type='test')

    # preprocess whole instances
    train_instances = preprocessor(train_instances, stopwords=True)
    test_instances = preprocessor(test_instances, stopwords=True)

    # preprocess separated instances
    for i in [anger_train, fear_train, joy_train, sad_train, anger_test, fear_test, joy_test, sad_test]:
        i = preprocessor(i, stopwords=True)

    # create ordered lists from whole instances
    tweets_train, sentiments_train, intensities_train = get_ordered_lists(train_instances)
    tweets_test, sentiments_test, intensities_test = get_ordered_lists(test_instances)

    # create ordered lists from separated instances
    tweets_anger_train, sentiments_anger_train, intensities_anger_train = get_ordered_lists(anger_train)
    tweets_fear_train, sentiments_fear_train, intensities_fear_train = get_ordered_lists(fear_train)
    tweets_joy_train, sentiments_joy_train, intensities_joy_train = get_ordered_lists(joy_train)
    tweets_sad_train, sentiments_sad_train, intensities_sad_train = get_ordered_lists(sad_train)

    tweets_anger_test, sentiments_anger_test, intensities_anger_test = get_ordered_lists(anger_test)
    tweets_fear_test, sentiments_fear_test, intensities_fear_test = get_ordered_lists(fear_test)
    tweets_joy_test, sentiments_joy_test, intensities_joy_test = get_ordered_lists(joy_test)
    tweets_sad_test, sentiments_sad_test, intensities_sad_test = get_ordered_lists(sad_test)

    # create csv vector files for whole tweet lists
    affective_vectorizer(tweets_train, 'train/whole_train.csv')
    affective_vectorizer(tweets_test, 'test/whole_test.csv')

    # create csv vector files for separated tweets
    affective_vectorizer(tweets_anger_train, 'train/anger.csv')
    affective_vectorizer(tweets_fear_train, 'train/fear.csv')
    affective_vectorizer(tweets_joy_train, 'train/joy.csv')
    affective_vectorizer(tweets_sad_train, 'train/sad.csv')

    affective_vectorizer(tweets_anger_test, 'test/anger.csv')
    affective_vectorizer(tweets_fear_test, 'test/fear.csv')
    affective_vectorizer(tweets_joy_test, 'test/joy.csv')
    affective_vectorizer(tweets_sad_test, 'test/sad.csv')

    # extract vectors from csv for whole datasets
    vect_train = extract_vectors('data/affect-vectors/train/whole_train.csv')
    vect_test = extract_vectors('data/affect-vectors/test/whole_test.csv')

    # extract vectors from csv for separated datasets
    anger_vect_train = extract_vectors('data/affect-vectors/train/anger.csv')
    fear_vect_train = extract_vectors('data/affect-vectors/train/fear.csv')
    joy_vect_train = extract_vectors('data/affect-vectors/train/joy.csv')
    sad_vect_train = extract_vectors('data/affect-vectors/train/sad.csv')

    anger_vect_test = extract_vectors('data/affect-vectors/test/anger.csv')
    fear_vect_test = extract_vectors('data/affect-vectors/test/fear.csv')
    joy_vect_test = extract_vectors('data/affect-vectors/test/joy.csv')
    sad_vect_test = extract_vectors('data/affect-vectors/test/sad.csv')
