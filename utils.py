import pandas as pd 
from sklearn.model_selection import train_test_split

def model_prep(dataset):

	categorical_columns=dataset.columns[dataset.dtypes=='object']
	numeric_columns=dataset.columns[dataset.dtypes!='object']
	encoded_data=pd.get_dummies(dataset,columns=categorical_columns)

	X=encoded_data.drop("exited",axis=1)
	y=encoded_data["exited"]
	x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

	return x_train,x_test,y_train,y_test


