# COMP 550 FINAL PROJECT


class EmoIntInstance:
    def __init__(self, id, tweet, sentiment, intensity):
        self.id = id
        self.tweet = tweet
        self.sentiment = sentiment
        self.intensity = intensity

    def __str__(self):
        return '%s\t%s\t%s\t%d' % (self.id, self.tweet, self.sentiment, self.intensity)


def create_instances(file):
    '''
    Transforms the dataset in a dict of EmoIntInstances
    :param file: Path of the dataset (str)
    :return: A dict of EmoIntInstances
    '''
    lines  = open(file).read().splitlines()
    instances = {}

    for line in lines:
        split_line = line.split('\t', 3)
        id = split_line[0]
        tweet = split_line[1]
        sentiment = split_line[2]
        intensity = float(split_line[3])
        instances[id] = EmoIntInstance(id, tweet, sentiment, intensity)

    return instances


def load_instances_separated(type='train'):
    '''
    Loads dicts of EmoIntInstance, with each emotion in a different dict.
    :param type: 'train' or 'test'. The type of dataset to load.
    :return: 4 dicts of EmoIntInstance. 1 for each emotion.
    '''
    assert type == 'train' or type == 'test', 'Invalid instance type, try train/test'

    if type == 'train':
        anger_train = create_instances('data/train/anger-ratings-0to1.train.txt')
        fear_train = create_instances('data/train/fear-ratings-0to1.train.txt')
        joy_train = create_instances('data/train/joy-ratings-0to1.train.txt')
        sad_train = create_instances('data/train/sadness-ratings-0to1.train.txt')

        anger_dev = create_instances('data/dev/anger-ratings-0to1.dev.gold.txt')
        fear_dev = create_instances('data/dev/fear-ratings-0to1.dev.gold.txt')
        joy_dev = create_instances('data/dev/joy-ratings-0to1.dev.gold.txt')
        sad_dev = create_instances('data/dev/sadness-ratings-0to1.dev.gold.txt')

        # merge dictionaries (requires python 3.5+)
        anger_instances = {**anger_train, **anger_dev}
        fear_instances = {**fear_train, **fear_dev}
        joy_instances = {**joy_train, **joy_dev}
        sad_instances = {**sad_train, **sad_dev}
    else:
        anger_instances = create_instances('data/test/anger-ratings-0to1.test.gold.txt')
        fear_instances = create_instances('data/test/fear-ratings-0to1.test.gold.txt')
        joy_instances = create_instances('data/test/joy-ratings-0to1.test.gold.txt')
        sad_instances = create_instances('data/test/sadness-ratings-0to1.test.gold.txt')

    return anger_instances, fear_instances, joy_instances, sad_instances


def load_instances():
    '''
    loads the datasets without separating the emotions.
    :return: 2 dicts of EmoIntInstance, one for the training set and one for the testing set.
    '''
    anger_train = create_instances('data/train/anger-ratings-0to1.train.txt')
    fear_train = create_instances('data/train/fear-ratings-0to1.train.txt')
    joy_train = create_instances('data/train/joy-ratings-0to1.train.txt')
    sad_train = create_instances('data/train/sadness-ratings-0to1.train.txt')

    anger_dev = create_instances('data/dev/anger-ratings-0to1.dev.gold.txt')
    fear_dev = create_instances('data/dev/fear-ratings-0to1.dev.gold.txt')
    joy_dev = create_instances('data/dev/joy-ratings-0to1.dev.gold.txt')
    sad_dev = create_instances('data/dev/sadness-ratings-0to1.dev.gold.txt')

    anger_test = create_instances('data/test/anger-ratings-0to1.test.gold.txt')
    fear_test = create_instances('data/test/fear-ratings-0to1.test.gold.txt')
    joy_test = create_instances('data/test/joy-ratings-0to1.test.gold.txt')
    sad_test = create_instances('data/test/sadness-ratings-0to1.test.gold.txt')

    # merge dictionaries (requires python 3.5+)
    train_instances = {**anger_train, **fear_train, **joy_train, **sad_train}
    dev_instances = {**anger_dev, **fear_dev, **joy_dev, **sad_dev}
    test_instances = {**anger_test, **fear_test, **joy_test, **sad_test}

    train_instances = {**train_instances, **dev_instances}

    return train_instances, test_instances
