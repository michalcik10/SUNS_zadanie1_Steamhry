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

test_data = pd.read_csv('C:/Users/micha/Desktop/SUNS/zadanie1/Z1-data/test.csv')
train_data = pd.read_csv('C:/Users/micha/Desktop/SUNS/zadanie1/Z1-data/train.csv')
#vypis velkosti dat
print(test_data.size)
print(train_data["score"].size)

#db = pd.DataFrame(train_data)
#print(test_data.isna().sum())

train_data= train_data.drop(columns=['VYMAZAT_price'])
train_data= train_data.drop(columns=['D_appid'])
print(train_data.isna().sum())
#uprava hodnoty score prázdne hodnoty negative nahradíme 0.0
train_data['score'] = train_data['score'].fillna(0.0)

#spracovanie datumu
train_data['D_release_date'] = pd.DatetimeIndex(train_data['D_release_date']).year


#print(train_data['D_release_date'])
#nahradenie nan hodnot priemernou hodnotou roku v ktorom su vydane hry
without_none_val = train_data['D_release_date'].dropna()
avarage_year = round(sum(without_none_val)/len(without_none_val))

train_data['D_release_date'] = train_data['D_release_date'].replace(np.nan,avarage_year)
#print(train_data.isna().sum())
#for index, row in train_data.iterrows():
#print(train_data['is_free'].dtype)
skuska = train_data.replace({True: 1, False: 0})
#print(skuska['self_published'])
#print(train_data.isna().sum())



#print(test_data.size)

#print(train_data.rows)

for column in (train_data.columns):
    print("{} ({:.3f}%)\fnull values in {}\t".format(train_data[column].isnull().sum(), train_data[column].isnull().sum()*100/len(train_data),column))