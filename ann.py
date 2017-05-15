import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,\
                                  OneHotEncoder,\
                                  StandardScaler
from sklearn.model_selection import train_test_split,\
                                    StratifiedKFold,\
                                    GridSearchCV
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

# Read data
df = pd.read_csv('Churn_Modeling.csv')
X = df.iloc[:,3:-1].values
y = df.iloc[:,-1].values

# Encode categorical
le_geography = LabelEncoder()
X[:,1] = le_geography.fit_transform(X[:,1])

le_gender = LabelEncoder()
X[:,2] = le_gender.fit_transform(X[:,2])

onehot = OneHotEncoder(categorical_features=[1])
X = onehot.fit_transform(X).toarray()
X = X[:,1:] # Dummy variable trap

# Split train and test
X_train, X_test, y_train, y_test = \
        train_test_split(X,y,test_size=0.2,random_state=28)

# Standardize
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Build network w/ Keras
def build_network(nodes,dropout):

    # Initialize
    model = Sequential()

    # Add layers
    model.add(Dense(nodes,\
                    activation='relu',\
                    kernel_initializer='truncated_normal',\
                    bias_initializer=keras.initializers.Constant(0.1),\
                    input_dim=11))
    model.add(Dropout(dropout))
    model.add(Dense(nodes,\
                    activation='relu',\
                    kernel_initializer='truncated_normal',\
                    bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dropout(dropout))
    model.add(Dense(1,\
                    activation='sigmoid',\
                    kernel_initializer='truncated_normal',\
                    bias_initializer=keras.initializers.Constant(0.1)))

    # Learning configurations
    model.compile(optimizer='adam',\
                  loss='binary_crossentropy',\
                  metrics=['accuracy'])

    return model

# Wrapper for sklearn functionality
model = KerasClassifier(build_network,batch_size=100,epochs=75)

# Hyperparameters to test w/ cross validation (2*2*8=32 fits)
# Add more if desired
test_params = {'nodes':[32,64],\
               'dropout':[0.25,0.5]}

skfold = StratifiedKFold(n_splits=8,shuffle=True,random_state=28)
grid_search = GridSearchCV(model,test_params,cv=skfold)
grid_search.fit(X_train,y_train)

best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Run on test
model = build_network(nodes=64,dropout=0.5)
model.fit(X_train,y_train,batch_size=100,epochs=75)
y_pred = model.predict_classes(X_test)

# Confusion matrix
cm = confusion_matrix(y_test,y_pred)

# Test accuracy
accuracy = (cm[0,0]+cm[1,1])/len(X_test)

# Ex.
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# Does this customer have a credit card: Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $50000

# Will this customer leave the bank?

customer = np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])
customer = ss.transform(customer)
leave_prob = model.predict(customer)

# This customer is very unlikely to leave