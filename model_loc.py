import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("new_data.csv")
data_model = data


#c = data['shelter_id'].astype('category')
#d = dict(enumerate(c.cat.categories))

#data_model['shelter_id'] = data_model['shelter_id'].astype('category').cat.codes
X = data_model[['outcome_lng','outcome_lat']]
Y = data_model[['found_lng','found_lat']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=142)
KNN_model = KNeighborsRegressor(n_neighbors=1).fit(X_train,Y_train)
#KNN_predict = KNN_model.predict(X_test)
#print(KNN_predict)
pickle.dump(KNN_model,open('model_location.pkl','wb'))