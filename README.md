# Predicting-House-Sale-prices
Machine Learning model
import pandas  as pd
import utilss
from sklearn.ensemble import RandomForestClassifier

train= pd.read_csv("train.csv")
utilss.clean(train)

test = pd.read_csv("test.csv")
utilss.clean(test)
test_X = test[["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","MasVnrArea"]].values

train_Y = train["SalePrice"].values
train_X = train[["MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt","MasVnrArea"]].values
val_model = RandomForestClassifier(n_estimators=500)
val_model.fit(train_X,train_Y)
pred=val_model.predict(test_X)
