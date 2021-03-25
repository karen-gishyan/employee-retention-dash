### Construct independent elements.
### Organize them into tabs, then tabs into layouts. or strictly into layouts.
### This approach allows to easiliy take out and modify graphs, and manually inserting
### each plot into the layout makes the complicated.

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import sys 
import dash
import dash_core_components as dcc
import dash_html_components as html 
import dash_bootstrap_components as dbc
import warnings
import plotly.express as px 
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from utils import *
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import pydotplus
import graphviz
from IPython.display import Image
from io import StringIO

warnings.filterwarnings("ignore")

data=pd.read_csv("Churn_Modelling.csv")

columns_in_lowercase=[]

for index in range(len(data.columns)):
	columns_in_lowercase.append(data.columns[index].lower())


data.columns=columns_in_lowercase
data.drop(["rownumber","customerid","surname"],axis=1,inplace=True)
#print(data.columns)

churn=data.query('exited==1')
remain=data.query('exited==0')


target_col=["exited"]

#selects the categorical columns if the number of unique elements is less than 6.
categorical_cols=data.nunique()[data.nunique() < 6].keys().tolist() #keys() selects the indices.

categorial_cols=[col for col in categorical_cols if col not in target_col]
numerical_cols=[col for col in data.columns if col not in categorical_cols+target_col]

#Statistical Properties.
data[data.columns[:10]].describe()

data[data.columns[:10]].median()

percentage_labels=data[target_col[0]].value_counts(normalize=True)*100
	
sns.set_theme(style="ticks",font_scale=1)
ax=sns.countplot(data.exited)
ax.set_title("Distribution")
ax.set_xlabel("Number of People who stayed /exited")
ax.set_ylabel("Values")

total_length=len(data['exited'])
for p in ax.patches:
	
	height=p.get_height()
	ax.text(p.get_x()+p.get_width()/2., height+2,f"{100*height/total_length}")

#plt.show()

### Three main steps.
### Make the chart, with a callback, embed in an html form.


app=dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

#each div has its own dropdown and graph. Later, a more effective method may be found.
popover_content=[dbc.PopoverHeader("Make a Ridge Classifier prediction based on your selected parameters.")]

controls=dbc.Card([

	
	dbc.FormGroup([
		#dbc.Label("CreditScore: Range is from 300-1000"),
		html.P("CreditScore: Valid range (300-1000).",className="column-colors"),
		#dbc.Input(id="credit-input",type="number", min=300, max=1000, step=1,valid=True,placeholder="Enter a creditScore")]),
		dcc.Slider(id='credscore-input', min=300,max=1000,step=1,marks={300:'300',500:"500",700:"700",900:"900"},value=500)]),


	dbc.FormGroup([
		dbc.Label("Geography:",className="column-colors"),
		dcc.Dropdown(id="geography-input",
		options=[{"label":value,"value":value} for value in data.geography.unique()],
		value="France")
		]),

	dbc. FormGroup([
		dbc.Label("Gender:",className="column-colors"),
		dbc.RadioItems(options=[{"label":"Male","value":1}, {"label":"Female","value":2}],
			value=1, id="gender-input",inline=True)]),

	dbc. FormGroup([
		dbc.Label("Age: Valid range (16-120).",className="column-colors"),
		dbc.Input(id="age-input",type="number", min=16, max=120, step=1,valid=True,placeholder="Enter a creditScore",value=20)]),


	dbc. FormGroup([
		dbc.Label("Length of Stay in Company: Valid range (0-50).",className="column-colors"),
		dbc.Input(id="tenure-input",type="number", min=0, max=50, step=1,valid=True,
			value=5,placeholder="Enter a creditScore")]),

	dbc. FormGroup([
		dbc.Label("Balance: Valid range (0-10,000,000).",className="column-colors"),
		dbc.Input(id="balance-input",type="number", min=0, max=10000000, step=1,
			valid=True,value=77000,  placeholder="Enter a creditScore")]),

	# dbc. FormGroup([
	# 	dbc.Label("Number of Products: Valid range (1-5).",className="column-colors"),
	# 	dbc.Input(id="num-product-input",type="number", min=0, max=10, step=1,
	# 		valid=True,value=2)]),


	dbc. FormGroup([
		dbc.Label("CreditCard Availability:",className="column-colors"),
		dbc.Checklist(id='credcard-input',
			options=[{"label":"No","value":1},{"label":"Yes","value":2}],
			value=[2],switch=True)]),

	# dbc. FormGroup([
	# 	dbc.Label("Active Membership:",className="column-colors"),
	# 	dbc.RadioItems(options=[{"label":"No","value":1}, {"label":"Yes","value":2}],
	# 		value=1, id="membership-input",inline=True)]),

	dbc.FormGroup([
		dbc.Label("Salary: Valid range (1,000-10,000,000).",className="column-colors"),
		dbc.Input(id="salary-input",type="number", min=1000, max=10000000, step=1,
			valid=True,placeholder="Enter a creditScore",value=100000)]),

	dbc.Button("Make a prediction", color="success", className="mr-1",id="predict-button"),
		dbc.Popover(popover_content,id='click',target='predict-button',trigger='click'),
		dbc. FormGroup([
					
	dbc.Label("Prediction Result",className="column-colors",style={'text-align': 'center'}),
	dbc.Input(id="predicted-status")])



	])
	

hist=html.Div([

	html.P("Mean:"),
	dcc.Slider(id="mean",min=-3,max=3,value=0,marks={-3:'-3','3':'3'}),
	html.P("Standard Deviation:"),
	dcc.Slider(id="std",min=1,max=5,value=1,marks={1:'1',5:'5'}),
	dcc.Graph(id="histogram")]	
	,className="card-hist")

pie=html.Div([

	html.P("Names:"),
	dcc.Dropdown(id="names",
	options=[{'label':x,'value':x} for x in data.columns],
	value="geography"),
	dcc.Graph(id="pie-chart")], className='card-pie')


table=html.Div([
	dbc.Table.from_dataframe(data.iloc[:5,:],striped=True,bordered=True,hover=True)
	],className="subheader-title")


tab1_content = dbc.Card(
	dbc.CardBody(
		[
		html.P("Employee Data",className="subheader-title"),
		table, 
		html.P("Statistical Charts",className="subheader-title"),
		dbc.Row([dbc.Col(hist),dbc.Col(pie)])]),  className="mt-3")


tab2_content = dbc.Card(
	dbc.CardBody(
		[
		html.P("Ridge Classifier",style={'color': '#008B8B', 'fontSize': 30,'font-weight': 'bold','text-align': 'center'}),
		dbc.Row([dbc.Col(controls),
			dbc.Col(dbc.Card([
				dcc.Graph(id="coef-figure"),

		]))],align="center"),
			
		]),className="mt-3")

tabs = dbc.Tabs([
		dbc.Tab(tab1_content, label="Data and Statistical Plots",disabled=False),
		dbc.Tab(tab2_content, label="Machine Learning")])

app.layout=dbc.Container([
	html.H1("Employee Retention Analysis",className='header-title'),
	tabs,
	html.Hr()
		
	],fluid=True)	
	

### the callback input goes into the function. Every chart needs its callback.

#State does not trigger callback, while Input does, so changing age, cred-score
@app.callback(
	[Output("coef-figure",component_property="figure"),
	Output("predicted-status",component_property="value")],
	
	Input("predict-button","n_clicks"),
	State("credscore-input","value"),
	State("age-input","value"),	
	State("tenure-input","value"),
	State("balance-input","value"),
	State("credcard-input","value"),
	State("salary-input","value"),
	State("geography-input","value"),
	State("gender-input","value")	
	)

### need to match the input order.
def ridge_classification(predict_button,cred_score,age,tenure,
	balance,credcard,salary, geography, gender):

	np.set_printoptions(suppress=True)

	x_train,x_test,y_train,y_test=model_prep(data)

	model=RidgeClassifier()
	model.fit(x_train,y_train)
	pred=model.predict(x_test)
	test_mae=metrics.mean_absolute_error(y_test,pred)
	
	coefs=np.array(model.coef_).ravel()
	sorted_x=[x for x,_ in sorted(zip(x_test.columns,coefs), key=lambda pair: pair[1])]
	sorted_coefs=np.sort(coefs)
	
	result="No Prediction"
	if predict_button:			
		
		list_=np.array(x_test.iloc[0,:])
		list_[0]=cred_score
		list_[1]=age
		list_[2]=tenure
		list_[3]=balance
		list_[5]=credcard[0]
		list_[7]=salary

		if geography=="France":
			al=[1,0,0]
		elif geography=="Germany":
		
			al=[0,1,0]
		else:
			al=[0,0,1]

		for i,j in zip([8,9,10],al):
			list_[i]=j

		if gender=="Male":
			list_[11]=1
			list_[12]=0
		
		else:
			list_[11]=0
			list_[12]=1	

		result=model.predict(list_.reshape(1,-1))
			
		# a=np.array(prediction_list).reshape((1,-1))
		# result=model.predict(a)

	coef_fig = px.bar(
		y=sorted_coefs,
		x=sorted_x,
		orientation="v",
		#color=x_test.columns.isin(numeric_columns),
		labels={"x": "Weight on Prediction", "y": "Features", "color": "Is numerical"},
		title="Ridge Classifier Feature Importance",
	)

	actual_v_pred_fig=px.scatter(x=y_test,y=pred,labels={"x":"Actual","y":"Predicted"},
		title=f"Actual vs Predicted with MAE {test_mae}")

	
	status="Stays" if result==0 else "Leaves"

	return coef_fig, status


def decision_tree_classification():

	x_train,x_test,y_train,y_test=model_prep(data)

	model=DecisionTreeClassifier(max_depth=3)
	model.fit(x_train,y_train)


@app.callback(
	Output('histogram','figure'),
	Input("mean","value"),Input("std","value"))

def make_histogram(mean,std):

	data=np.random.normal(mean,std,size=800)
	fig=px.histogram(data,nbins=10)
	return fig


@app.callback(
	Output('pie-chart','figure'),
	Input("names","value"))

def make_pie_chart(names):
	fig=px.pie(data,values=data.age,names=names)
	return fig 


if __name__ == "__main__":

	app.run_server(debug=True)




