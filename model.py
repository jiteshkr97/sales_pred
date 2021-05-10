
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error


walmart=pickle.load(open('train_data','rb')) 


walmart

# creating OHE for isholiday feature
walmart['IsHoliday'] = pd.get_dummies(walmart['IsHoliday'],drop_first=True)


train=walmart.copy()

train.columns

# filling null value with 0
train=train.fillna(0)

# drop unnecceassy features
train = train.drop(['Date','Temperature', 'Fuel_Price','MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI','Unemployment', 'Type','month'],axis=1)

#scaling the size feature
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train["Size"].values.reshape(-1,1))  #fit has to be done only on Train data
md5_train = scaler.transform(train["Size"].values.reshape(1,-1))

train['Size'] = md5_train.reshape(-1,1)

train['IsHoliday'] = pd.get_dummies(train['IsHoliday'],drop_first=True)

# # Checking best hyperparameters on tarin data

X=train.drop('Weekly_Sales',axis=1)
X.shape

y=train['Weekly_Sales']
y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# applying ohe on isholiday feature
train['IsHoliday'] = pd.get_dummies(train['IsHoliday'],drop_first=True)

# checking for xg boost model
from xgboost import XGBRegressor
rf=XGBRegressor( n_estimators=360, max_depth=10)
rf.fit(X_train,y_train)

# Saving model to disk
pickle.dump(rf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
pred = model.predict(X_test)

#check the accuracy
mean_absolute_error(y_test,pred)
