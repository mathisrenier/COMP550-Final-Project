class Tweet(object):

    def __init__(self, id, text, emotion, intensity):
        self.id = id
        self.text = text
        self.emotion = emotion
        self.intensity = intensity


def read_data(path):

    list = list()
    with open(path) as input_file:
        for line in input_file:
            line = line.strip()
            array = line.split('\t')
            train_list.append(Tweet(array[0], clean_str(array[1]), array[2], float(array[3])))
    return list



def evaluate(pred, gold):
    if len(pred) == len(gold):

        
        # return zero correlation if predictions are constant
        if np.std(pred)==0 or np.std(gold)==0:
            return (0,0,0,0)
        
        pears_corr=st.pearsonr(pred,gold)[0]                                    
        spear_corr=st.spearmanr(pred,gold)[0]   



      
        return np.array([pears_corr,spear_corr])













x_train = vectorize_tweets(train)
x_test = vectorize_tweets(test)




x_train = sequence.pad_sequences(x_train, maxlen=embedding_info[1], padding="post", truncating="post", 
                                 dtype='float64')
x_test = sequence.pad_sequences(x_test, maxlen=embedding_info[1], padding="post", truncating="post",
                                dtype='float64')




conv_1 = Conv1D(300, conv_kernel, activation='relu', input_shape=x_train.shape[1:])
pool_1 = MaxPooling1D()
flat_1 = Flatten()
dense_1 = Dense(30000, activation='relu')
dense_2 = Dense(1, activation='sigmoid')
drop_1 = Dropout(rate=0.5)




def getModel():
    
    model = Sequential()

    model.add(conv_1)
    model.add(pool_1)
    

    model.add(flat_1)
    
    model.add(dense_1)
    model.add(drop_1)
    model.add(dense_2)

    model.compile(loss='mean_squared_error', optimizer="sgd")
    
    return model



estimator = KerasRegressor(build_fn=getModel, epochs=200, batch_size=16, verbose=1)
ml_model.fit(x_train, score_train)
y_pred = ml_model.predict(x_test)
y_pred = y_pred.reshape((760,))


evaluate(y_pred, y_gold)