# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# Any results you write to the current directory are saved as output.

from keras.models import Model,Sequential
from keras.layers import Dense, Embedding, Input,Activation,BatchNormalization
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from keras import backend as K


max_features = 30000
maxlen = 512


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#train = train.sample(frac=1) return a random sample of items 

list_sentences_train = train["comment_text"].fillna("unknown").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("unknown").values


tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

'''
def myloss(y_true,y_pred): 
	#binary_crossentropy
	result = []
    for i in range(len(y_pred)):
        y_pred[i] = [max(min(x, 1 - K.epsilon()), K.epsilon()) for x in y_pred[i]]
        result.append(-np.mean([y_true[i][j] * math.log(y_pred[i][j]) + (1 - y_true[i][j]) * math.log(1 - y_pred[i][j]) for j in range(len(y_pred[i]))]))
    return np.mean(result)
'''

def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[])

    return model

model = get_model()

'''
model =Sequential()
model.add(Embedding(max_features,embed_size,input_length=maxlen))
model.add(LSTM(64, return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(6, activation="sigmoid"))
model.compile(loss='binary_crossentropy',
                  optimizer='SGD',
                  metrics=[])
'''

batch_size = 256
epochs = 10


file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

logdir="graph"
board=TensorBoard(logdir,histogram_freq=0,write_graph=True,write_images=True)

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, board, early] #early

model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

#model.load_weights(file_path)

y_test = model.predict(X_te)



sample_submission = pd.read_csv("sample_submission.csv")

sample_submission[list_classes] = y_test



sample_submission.to_csv("QILIN.csv", index=False)