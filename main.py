import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as pg
import plotly.figure_factory as ffc
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks

from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# FUNCTIONS #
# fill empty cells in all data (use only if all data are prepare)
def NullEmptyValues(df):
    for col in df.columns:
        print("{} ({:.2f})\tnull values in {}\t".format(df[col].isnull().sum(),
                                                        df[col].isnull().sum() * 100 / len(df), col))
    print("")


# normalisation of data in one collum
def normalisation(df):
    for col in df.columns:
        if col == 'Potability':
            break
        max_val = df[col].max()
        min_val = df[col].min()
        df[col] = [(row - min_val) / (max_val - min_val) for row in df[col]]


# prepare test data
test_data = pd.read_csv('water_test.csv')
# remove Sulfate collum cause is unused
test_data = test_data.drop(columns=['Sulfate'])
test_data = test_data.fillna(0)
normalisation(test_data)

# read data
data = pd.read_csv('water_train.csv')

# print number of rows in csv
print("{} items".format(len(data)))

# check null or empty cols in rows / csv data checking
NullEmptyValues(data)

# remove rows with empty PH value
data.dropna(subset=['ph'], inplace=True)

# remove Sulfate collum cause is unused
data = data.drop(columns=['Sulfate'])

# fill Trihalomethanes null or empty elements with 0
data = data.fillna(0)

# check null or empty cols in rows / csv data checking
NullEmptyValues(data)

# mean and standard deviation of data
print(data.describe())
fig = px.histogram(data, x='Solids')
#fig.show()

# normalisation of data
normalisation(data)

# mean and standard deviation of data
print(data.describe())
fig = px.histogram(data, x='Solids')
#fig.show()

# set train, valid and test data
train_data = data.sample(frac=0.8)
valid_data = data.drop(train_data.index)

x_train = train_data.drop(columns='Potability')
y_train = train_data['Potability']

x_valid = valid_data.drop(columns='Potability')
y_valid = valid_data['Potability']

x_test = test_data.drop(columns='Potability')
y_test = test_data['Potability']

# prepare y labels for neural network
y_train_labels = utils.to_categorical(y_train)
y_valid_labels = utils.to_categorical(y_valid)

# logistic regression test data
lgr = LogisticRegression(random_state=0, max_iter=1000).fit(x_train, y_train.values.ravel())
lgr_predict = lgr.predict(x_test)
print(classification_report(y_test, lgr_predict))

# neural network part
# set model, layers ...
callback = callbacks.EarlyStopping(monitor='val_loss', patience=10)

best_per = 0
best_layers_neurone = 0
best_layers_num = 0
best_learning_rate = 0
for layers_neurone in range(1, 100):
    print('Layer neurones->' + str(layers_neurone))
    model = models.Sequential()
    model.add(layers.Dense(layers_neurone, input_shape=(x_train.shape[1],), activation='relu'))
    for layers_num in range(1, 5):
        print('Hidden layers->\t' + str(layers_num))
        for num_l in range(1, layers_num):
            model.add(layers.Dense(layers_neurone, activation='relu'))
        model.add(layers.Dense(2, activation='sigmoid'))

        learning_rate = 0.00005
        while learning_rate < 0.0005:
            print("Learning rate->\t" + str(learning_rate))
            adam = optimizers.Adamax(learning_rate=0.0005)

            # train neural network
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics='accuracy')
            training = model.fit(x_train, y_train_labels, epochs=400, callbacks=[callback], verbose=0, validation_data=(x_valid, y_valid_labels))


            # print training data to graph
            # training_history = pd.DataFrame.from_dict(training.history)
            # print(training_history)
            #
            # fig_train_his = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy", 'Loss'])
            # fig_train_his.add_trace(pg.Scatter(y=training_history['loss'], name='loss', mode='lines'), row=1, col=2)
            # fig_train_his.add_trace(pg.Scatter(y=training_history['val_loss'], name='val_loss', mode='lines'), row=1, col=2)
            # fig_train_his.add_trace(pg.Scatter(y=training_history['accuracy'], name='accuracy', mode='lines'), row=1, col=1)
            # fig_train_his.add_trace(pg.Scatter(y=training_history['val_accuracy'], name='val_accuracy', mode='lines'), row=1, col=1)
            #
            # fig_train_his.show()

            # neural network prediction on test data
            y_predict_test = np.argmax(model.predict(x_test), axis=1)
            # classify = classification_report(y_test.values, y_predict_test)
            score = accuracy_score(y_test, y_predict_test, normalize=True)
            print('Score->\t\t\t' + str(score))
            if best_per < score:
                best_per = score
                best_layers_neurone = layers_neurone
                best_layers_num = layers_num
                best_learning_rate = learning_rate

            learning_rate = learning_rate + 0.00005
        print('Best score->\t\t' + str(best_per))
    print('{} Best s, {} lne, {} lnu, {} lr'.format(best_per, best_layers_neurone, best_layers_num, best_learning_rate))
# confusion matrix
# matrix = confusion_matrix(y_test.values, y_predict_test)
# matrix = np.flip(matrix, axis=0)
#
# matrix_text = [[str(y) for y in x] for x in matrix]
# fig_matrix = ffc.create_annotated_heatmap(matrix, ['Positive', 'Negative'], ['Negative', 'Positive'], matrix_text, colorscale='Viridis')

#fig_matrix.show()

