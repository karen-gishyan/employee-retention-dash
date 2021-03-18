#https://www.kaggle.com/bandiang2/prediction-of-customer-churn-at-a-bank

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
import sys 
import dash
import dash_core_components as dcc 
import dash_html_components as html 
import warnings
import plotly.express as px 
from dash.dependencies import Input, Output
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

app=dash.Dash(__name__)

#each div has its own dropdown and graph. Later, a more effective method may be found.
app.layout=html.Div([
	html.Div([
	html.P("Names:"),
	dcc.Dropdown(id="names",
		options=[{'label':x,'value':x} for x in data.columns],
		value="geography"),
	dcc.Graph(id="pie-chart") 
	]),
	html.Div([
	html.P("Mean:"),
	dcc.Slider(id="mean",min=-3,max=3,value=0,marks={-3:'-3','3':'3'}),
	html.P("Standard Deviation:"),
	dcc.Slider(id="std",min=1,max=5,value=1,marks={1:'1',5:'5'}),
	dcc.Graph(id="histogram")
	])
])


@app.callback(
	Output('pie-chart','figure'),
	[Input("names","value")])

def make_pie_chart(names):
	fig=px.pie(data,values=data.age,names=names)
	return fig 

@app.callback(
	Output('histogram','figure'),
	[Input("mean","value"),Input("std","value")])

def make_histogram(mean,std):
	data=np.random.normal(mean,std,size=800)
	fig=px.histogram(data,nbins=10)
	return fig


if __name__=="__main__":
	app.run_server(debug=True)



