# COMP 550 FINAL PROJECT

import loader
import nltk
import sys
import string
import weka.core.jvm as jvm
import weka.core.packages as packages



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
        tokens = nltk.word_tokenize(v.tweet.lower())

        if stemming:
            stemmer = nltk.stem.PorterStemmer()
            processed_tokens = [stemmer.stem(w) for w in tokens if w not in sw]
        else:
            lemmatizer = nltk.stem.WordNetLemmatizer()
            processed_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in sw]

        v.tweet = ' '.join(w for w in processed_tokens)
        tweets.append(v.tweet)
        sentiments.append(v.sentiment)
        intensities.append(v.intensity)

    return instances


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
    jvm.start(packages=True)

    install_package('AffectiveTweets')
    anger_train = loader.load_weka_data('data/weka-format/anger-ratings-0to1.train.dev.arff')

    jvm.stop()

    train_instances, dev_instances, test_instances = loader.load_instances()

    train_instances = preprocessor(train_instances, stopwords=True)
    tweets_train, sentiments_train, intensities_train = get_ordered_lists(train_instances)
