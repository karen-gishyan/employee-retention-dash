### Make your functions (for visualizations, models) then decorate with inputs and outputs.


### Construct independent elements.
### Organize them into tabs, then tabs into layouts. or strictly into layouts.
### This approach allows to easiliy take out and modify graphs, and manually inserting
### each plot into the layout makes the complicated.


import pandas as pd 
import numpy as np 
import sys 
import dash
import dash_core_components as dcc
import dash_html_components as html 
import dash_bootstrap_components as dbc
import warnings
import plotly.express as px 
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from utils import *

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
	
# sns.set_theme(style="ticks",font_scale=1)
# ax=sns.countplot(data.exited)
# ax.set_title("Distribution")
# ax.set_xlabel("Number of People who stayed /exited")
# ax.set_ylabel("Values")

# total_length=len(data['exited'])
# for p in ax.patches:
	
# 	height=p.get_height()
# 	ax.text(p.get_x()+p.get_width()/2., height+2,f"{100*height/total_length}")

### Three main steps.
### Make the chart, with a callback, embed in an html form.

svm_regression_X=data[['creditscore', 'age', 'balance']]
svm_regression_y=data['estimatedsalary']

app=dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
server=app.server


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


	dbc. FormGroup([
		dbc.Label("CreditCard Availability:",className="column-colors"),
		dbc.Checklist(id='redcard-input',
			options=[{"label":"No","value":1},{"label":"Yes","value":2}],
			value=[2],switch=True)]),

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
	

table=html.Div([
	dbc.Table.from_dataframe(data.iloc[:5,:],striped=True,bordered=True,hover=True)
	],className="subheader-title")


dropdown_items=[dbc.DropdownMenuItem(col) for col in numerical_cols]

hist=html.Div([

	dbc.Label("Variable:",className="column-colors"),
	dcc.Dropdown(id="hist-dropdown-items",
	options=[{"label":value,"value":value} for value in numerical_cols],
	value="age"),
	html.Hr(),

	#dbc.DropdownMenu(dropdown_items,id="hist-dropdown-items",label="Histogram Columns", color="primary"),
	dcc.Graph(id="histogram")],className="card-hist")

pie_1=dbc.FormGroup([
	
	dbc.Label("Names:",className="column-colors"),
	dcc.Dropdown(id="pie-chart-names",options=[{'label':x,'value':x} for x in categorial_cols],value="geography")])

pie_2=dbc.FormGroup([

	dbc.Label("Values:",className="column-colors"),
	dcc.Dropdown(id="pie-chart-values",options=[{'label':x,'value':x} for x in numerical_cols],value="age")])

pie_graph=dcc.Graph(id='pie-chart',className="card-pie")


pie=dbc.Col([pie_1,pie_2,pie_graph])
	
	
#numeric and categorical variables cannot be on the same axis at the same time.
#s
bar_chart=dbc.Card([
	dbc.Form([

	dbc.FormGroup([
	dbc.Label("x-axis:",className="column-colors"),
	
	dbc.RadioItems(id="bar-chart-x-axis",options=[{'value':col,'label':col}
		for col in ['geography','gender','isactivemember']],value='gender',inline=True)]),
	
	dbc.FormGroup([
	
	dbc.Label("y-axis:",className="column-colors"),
	dbc.RadioItems(id="bar-chart-y-axis",options=[{'value':col,'label':col}
	for col in ['age','creditscore','estimatedsalary','exited']],value='age',inline=True)])]),
	html.Hr(),
	
	dcc.Graph(id="box-plot",className="card-box")])


tab1_content = dbc.Card(
	dbc.CardBody(
		[
		html.P("Employee Data",className="subheader-title"),
		table, 
		html.P("Statistical Charts",className="subheader-title"),
		dbc.Row([dbc.Col(hist),pie]),
		html.Hr(),

	dbc.Row([
	dbc.Col(bar_chart)])]), className="mt-3") #Card wrapped in a Col, the other way round is also

#wrapping in Div is sometime better than wrapping in Card.
model_2=html.Div([
	html.Hr(),
	html.P("Train and Visualize yout custom Regression Model to predict Employee salary",style={'color': '#008B8B', 'fontSize': 30,'font-weight': 'bold','text-align': 'center'}),
	
	
	dbc.Row([
		dbc.Col([
	dbc.FormGroup([
	dbc.Label("Choose the first feature",className="column-colors"),
	dbc.RadioItems(id="feature-1",options=[{'value':col,'label':col}
		for col in svm_regression_X],value='age')])]),

	dbc.Col([
	dbc.FormGroup([
	dbc.Label("Choose the second feature",className="column-colors"),
	dbc.RadioItems(id="feature-2",options=[{'value':col,'label':col}
		for col in svm_regression_X],value='age')])]),
	]),
	html.Hr(),
	dbc.FormGroup([
		#dbc.Label("CreditScore: Range is from 300-1000"),
		html.P("Select the number of observations for trainin gyour model",className="column-colors"),
		#dbc.Input(id="credit-input",type="number", min=300, max=1000, step=1,valid=True,placeholder="Enter a creditScore")]),
		dcc.Slider(id='model-input', min=30,max=300,step=5,marks={100:'100',200:"300"},value=50)]),
	html.Hr(),
	dbc.FormGroup([
	dbc.Label("Please Select a Model for the Regression",className="column-colors"),	
	dbc.RadioItems(id="model_2_regressor",options=[{'value':col,'label':col}
		for col in ['Linear Regression','Decision Tree Regression','Support Vector Machine Regression','KNN Regression']],value='LinearRegression')]),
	
	dcc.Graph(id="svm-graph")])
		
tab2_content = dbc.Card(
	
	dbc.CardBody(
		[
		html.P("Ridge Classification for classifying whether the Employee will leave or stay",style={'color': '#008B8B', 'fontSize': 30,'font-weight': 'bold','text-align': 'center'}),
		dbc.Row([dbc.Col(controls),
			dbc.Card([
				dcc.Graph(id="coef-figure") # card can be wrapped in a column, no difference.

		])],align="center"),dbc.Row([dbc.Col(model_2)],align="center") # wrap al element in a col in a Row.
			
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
	#State("credcard-input","value"),
	State("salary-input","value"),
	State("geography-input","value"),
	State("gender-input","value")	
	)

### need to match the input order.
def ridge_classification(predict_button,cred_score,age,tenure,
	balance,salary, geography, gender):

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
		#list_[5]=credcard[0]
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

# @app.callback(
# 	Output("model_2_dropdown","value"),
# 	Input("model_2_dropdown","value"))

@app.callback(

	Output("svm-graph","figure"),
	[
	Input("feature-1","value"),
	Input("feature-2","value"),
	Input("model_2_regressor","value"),
	Input("model-input","value")
	])

def regression_plot(feature_1,feature_2,regression_model,n_rows):

	margin=0
	feature_list=['creditscore','balance']

	if feature_1 in feature_list and feature_2 in feature_list:
		step_size=50

	step_size=10
	data_filter=data[data.balance!=0]
	data_filter=data_filter.iloc[:500,:]
	X=data_filter.loc[:n_rows,[feature_1,feature_2]]	
	y=svm_regression_y[:X.shape[0]]

	if regression_model=='Linear Regression':
		model=LinearRegression()
	
	elif regression_model=='Decision Tree Regression':
		model=DecisionTreeRegressor()
	elif regression_model=='Support Vector Machine Regression':
		model=SVR()
	else:
		model=KNeighborsRegressor()

	model.fit(X,y)

	col1=X.iloc[:,0]
	col2=X.iloc[:,1]

	xmin,xmax=col1.min()-margin, col1.max()+margin
	ymin,ymax=col2.min()-margin, col2.max()+margin

	x_range=np.arange(xmin,xmax,step_size)
	y_range=np.arange(ymin,ymax,step_size)
	
	#understand meshgrid.
	xx,yy=np.meshgrid(x_range,y_range)

	pred=model.predict(np.c_[xx.ravel(),yy.ravel()])
	pred=pred.reshape(xx.shape)

	fig=px.scatter_3d(data_filter,x=col1.name,y=col2.name,z="estimatedsalary")
	#fig.update_traces(marker=dict(size=5))
	fig.add_trace(go.Surface(x=x_range,y=y_range,z=pred,name="surface"))
	
	return fig

@app.callback(
	Output('histogram','figure'),
	Input('hist-dropdown-items','value'))

def histogram(colname):

	fig=go.Figure()
	fig.add_trace(go.Histogram(x=churn[colname],histnorm='percent',name='Churning Customers',
		opacity=0.85))
	fig.add_trace(go.Histogram(x=remain[colname],histnorm='percent',name='Remaining Customers',
		opacity=0.85))
	#https://plotly.com/python/reference/layout/.
	fig.update_layout(title=dict(text=f"{colname.upper()} distribution according to customers",font_family="Balto"),
		xaxis=dict(title=colname,ticklen=10,gridwidth=3),yaxis=dict(title="percent",ticklen=10,gridwidth=3))

	return fig

@app.callback(
	Output('pie-chart','figure'),
	[Input("pie-chart-values","value"),Input("pie-chart-names","value")])

def make_pie_chart(values,names):
	fig=px.pie(data,values=values,names=names)
	return fig 

@app.callback(Output('box-plot','figure'),
	[Input("bar-chart-x-axis",'value'),
	Input("bar-chart-y-axis",'value')])


def box_plot(x,y,dataset=data):
	fig=px.box(dataset,x=x,y=y)

	return fig  

if __name__ == "__main__":

	app.run_server(debug=True)




