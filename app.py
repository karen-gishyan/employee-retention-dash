#https://www.kaggle.com/bandiang2/prediction-of-customer-churn-at-a-bank
#Build incrementally.


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
from dash.dependencies import Input, Output
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics


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


# print(data.columns)

#plt.show()

### Three main steps.
### Make the chart, with a callback, embed in an html form.



app=dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

#each div has its own dropdown and graph. Later, a more effective method may be found.

controls=dbc.Card([
	dbc.FormGroup([
		dbc.Label("Geography"),
		dcc.Dropdown(id="geography",
		options=[{"label":value,"value":value} for value in data.geography.unique()],
		value="France")
		])])


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
		dcc.Graph(id="pie-chart") ], className='card-pie')


table=html.Div([
	html.P("Table:"),
	dbc.Table.from_dataframe(data.iloc[:5,:],striped=True,bordered=True,hover=True)
	],className="c")

app.layout=dbc.Container([
	html.H1("Employee Retention Analysis",className='header-title'),
	html.Hr(),
	dbc.Row(table),
	dbc.Row([dbc.Col(controls),dbc.Col(dcc.Graph(id="regression-graph"))],align="center"),	
	dbc.Row([
		dbc.Col(hist),
		dbc.Col(pie)]),
	],fluid=True)	
	

### the callback input goes into the function. Every chart needs its callback.
@app.callback(
	Output("regression-graph","figure"),
	[Input("geography","value")])

def regression_graph():
	pass

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




