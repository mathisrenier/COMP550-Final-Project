# COMP 550 FINAL PROJECT


class EmoIntInstance:
    def __init__(self, id, tweet, sentiment, intensity):
        self.id = id
        self.tweet = tweet
        self.sentiment = sentiment
        self.intensity = intensity

    def __str__(self):
        return '%s\t%s\t%s\t%d' % (self.id, self.tweet, self.sentiment, self.intensity)

def load_instances(file):
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


def main():
    anger_train = load_instances('data/train/anger-ratings-0to1.train.txt')
    fear_train = load_instances('data/train/fear-ratings-0to1.train.txt')
    joy_train = load_instances('data/train/joy-ratings-0to1.train.txt')
    sad_train = load_instances('data/train/sadness-ratings-0to1.train.txt')

    anger_dev = load_instances('data/dev/anger-ratings-0to1.dev.gold.txt')
    fear_dev = load_instances('data/dev/fear-ratings-0to1.dev.gold.txt')
    joy_dev = load_instances('data/dev/joy-ratings-0to1.dev.gold.txt')
    sad_dev = load_instances('data/dev/sadness-ratings-0to1.dev.gold.txt')

    anger_test = load_instances('data/test/anger-ratings-0to1.test.gold.txt')
    fear_test = load_instances('data/test/fear-ratings-0to1.test.gold.txt')
    joy_test = load_instances('data/test/joy-ratings-0to1.test.gold.txt')
    sad_test = load_instances('data/test/sadness-ratings-0to1.test.gold.txt')


    train_instances = {**anger_train, **fear_train, **joy_train, **sad_train}
    dev_instances = {**anger_dev, **fear_dev, **joy_dev, **sad_dev}
    test_instances = {**anger_test, **fear_test, **joy_test, **sad_test}
    return train_instances, dev_instances, test_instances


if __name__ == '__main__':
    main()