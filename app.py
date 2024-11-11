# -*- coding: utf-8 -*-

# Загрузим необходимые пакеты
import dash
import dash_core_components as dcc
import dash_html_components as html

from dash import Input, Output, State, callback
import plotly.graph_objects as go
import pandas as pd
import dash_leaflet as dl
import dash_leaflet.express as dlx
import geojson

from plotly.graph_objs import Scatter, Figure, Layout
import plotly
import json
import geopandas as gpd
import plotly.express as px

#model
from sklearn.preprocessing import LabelEncoder
import numpy as np
import catboost as cb
import pickle

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('sub_area_rus.csv')


#start map
fig = go.Figure()
gdf = gpd.read_file('mo.geojson')
output = pd.DataFrame({'NAME': gdf['NAME'],'color': np.where(gdf['NAME']=='Аэропорт', '10','1')})
output.to_csv('NAME.csv', index=False)
output.head()

name_df = pd.read_csv('NAME.csv',dtype={"NAME":str})

with open('mo.geojson', encoding="utf-8") as response:
	mo = json.load(response)
	

with open('./model.pkl','rb') as model_pkl:
	model=pickle.load(model_pkl)
	
with open('./classifier.pkl','rb') as classifier_pkl:
	classifier=pickle.load(classifier_pkl)


app.layout = html.Div(

children=[
    html.H2(children='Предсказание цены на квартиру по выбранным параметрам'),

    html.Div(children='''
        Карта муниципальных округов Москвы для выбора (sub_area).
    '''),

    dcc.Graph( id='sub_area_map',
	figure = fig
	),	

	    html.Label('Выбранная территория(sub_area)'),	
    dcc.Dropdown(
					id = 'sub_area',
                   options=[{'label': k, 'value': k} for k in list(df['sub_area_rus'].values)],
                    value='Аэропорт',
                    placeholder="Name"),
	
	    html.Label('Тип продукта (product_type)'),
    dcc.RadioItems(id='product_type',
        options=[
            {'label': 'Инвестиции', 'value': '1'},
            {'label': 'Владение', 'value': '2'},
        ],
        value='1'
    ),

		html.Label('Материал (material)'),
	html.Div(dcc.Dropdown(id='material', options=[1,2,3,4,5,6], value = 1)),
	    html.Label('Состояние(state)'),
  	html.Div(dcc.Dropdown(id='state', options=[1,2,3,4], value = 1)),	
	
		html.Label('Количество комнат (num_room)'),
	html.Div(dcc.Input(id='num_room', type='number', inputMode='numeric', min = 1, max = 10, value = 1)),
	    html.Label('Общая площадь(full_sq)'),
	html.Div(dcc.Input(id='full_sq', type='number', inputMode='numeric', min = 1, max = 6000, value = 40)),
		html.Label('Жилая площадь(life_sq)'),
	html.Div(dcc.Input(id='life_sq', type='number', inputMode='numeric', min = 1, max = 7500, value = 30)),
		html.Label('Площадь кухни (kitch_sq)'),
	html.Div(dcc.Input(id='kitch_sq', type='number', inputMode='numeric', min = 1, max = 100, value = 10)),

		html.Label('Этаж (floor)'),
	html.Div(dcc.Input(id='floor', type='number', inputMode='numeric', min = 1, max = 50, value = 3)),
		html.Label('Этажность (max_floor)'),
	html.Div(dcc.Input(id='max_floor', type='number', inputMode='numeric', min = 1, max = 50, value = 5)),
		
		html.Label('Расстояние до метро в км (metro_km_walk)'),
	html.Div(dcc.Input(id='metro_km_walk', type='number', inputMode='numeric', min = 0, max = 60, value = 3)),	
#		html.Label('Время до метро (metro_min_walk)'),
#	html.Div(dcc.Input(id='metro_min_walk', type='number', inputMode='numeric', min = 0, max = 713, value = 3)),
#		html.Label('Расстояние до метро на автомобиле (metro_km_avto)'),
#	html.Div(dcc.Input(id='metro_km_avto', type='number', inputMode='numeric', min = 0, max = 74, value = 7)),
#		html.Label('Время до метро на автомобиле (metro_min_avto)'),
#	html.Div(dcc.Input(id='metro_min_avto', type='number', inputMode='numeric', min = 0, max = 62, value = 5)),		
#		html.Label('Расстояние до Третьего транспортного кольца (ttk_km)'),
#	html.Div(dcc.Input(id='ttk_km', type='number', inputMode='numeric', min = 0, max = 100, value = 11)),	
	html.Br(),
	html.Br(),
	html.Br(),
	html.Br(),
	html.Br(),
	html.Br(),
	html.Br(),
	html.Br(),
	html.Br(),
	html.Br(),

    html.Button('Предсказать стоимость квартиры по выбранным параметрам', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic',
             children='Введите данные и нажмите на кнопку'),

	html.Br(),
	html.Br(),

		html.Label('Цена квартиры для предсказания расположения(price)'),
	html.Div(dcc.Input(id='price', type='number', inputMode='numeric', min = 0, max = 1000000000, value = 5000000)),
			 
    html.Button('Предсказать расположение квартиры по указанной цене', id='submit2-val', n_clicks=0),
    html.Div(id='container2-button-basic',
             children='Введите данные и нажмите на кнопку, при этом ранее выбранное местоположение игнорируется'),
		 
], style={'columnCount': 3})

@callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('num_room', 'value'),
    State('full_sq', 'value'),
	State('life_sq', 'value'),
	State('kitch_sq', 'value'),
	State('floor', 'value'),
	State('state', 'value'),
    State('max_floor', 'value'),
    State('material', 'value'),
    State('product_type', 'value'),
    State('sub_area', 'value'),
#	State('ttk_km', 'value'),
#   State('metro_min_avto', 'value'),
#	State('metro_km_avto', 'value'),
#   State('metro_min_walk', 'value'),
	State('metro_km_walk', 'value'),
    prevent_initial_call=True
)   
def update_output(n_clicks, num_room_value, full_sq_value, life_sq_value, kitch_sq_value,floor_value,state_value,max_floor_value, material_value, product_type_value, sub_area_value, #ttk_km_value, metro_min_avto_value, metro_km_avto_value, metro_min_walk_value, 
metro_km_walk_value):
		X_test_dict = {
		"num_room": num_room_value,
		"full_sq": np.log(full_sq_value +1),
		"life_sq": life_sq_value,
		"floor": floor_value,
		"state": state_value,
		"max_floor": max_floor_value, 
		"material": material_value,
		"kitch_sq": kitch_sq_value,
		"product_type_num": product_type_value,
 #      "ttk_km": ttk_km_value,
 #      "metro_min_avto": metro_min_avto_value,
 #		"metro_km_avto": metro_km_avto_value,
 #      "metro_min_walk": metro_min_walk_value,
 		"metro_km_walk": metro_km_walk_value,
		"sub_area_num": df[df['sub_area_rus']==sub_area_value].id
		}
		X_test = pd.DataFrame([X_test_dict])
		y_pred = np.exp(model.predict(X_test))-1
		return 'Стоимость квартиры:"{:,}" '.format(round(y_pred[0]))

@callback(Output('sub_area','value'),
		  Input('sub_area_map','clickData'))
def update_dropdown(clickData):
	if clickData is None:
		value = 'Аэропорт'
	else:
		value = clickData['points'][0]['location']
	return value
	
@callback(Output('sub_area_map','figure'),
		  Input('sub_area','value'))
def update_dropdown(value):
	name_df['color']=np.where(name_df['NAME']==value, '10','1')
	fig = px.choropleth_mapbox(
		name_df,
        geojson=mo, 
		color="color",	
        locations="NAME", featureidkey="properties.NAME",
        center={"lat": 55.61895, "lon": 37.30607}, zoom=7.5,
        range_color=[0, 20])
	fig.update_layout(mapbox_style="carto-positron")
	fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
	return fig	
	
@callback(
    Output('container2-button-basic', 'children'),
    Output('sub_area','value',allow_duplicate=True),	
    Input('submit2-val', 'n_clicks'),
    State('num_room', 'value'),
    State('full_sq', 'value'),
	State('life_sq', 'value'),
	State('kitch_sq', 'value'),
	State('floor', 'value'),
	State('state', 'value'),
    State('max_floor', 'value'),
    State('material', 'value'),
    State('product_type', 'value'),
    State('price', 'value'),
#	State('ttk_km', 'value'),
#   State('metro_min_avto', 'value'),
#	State('metro_km_avto', 'value'),
#   State('metro_min_walk', 'value'),
	State('metro_km_walk', 'value'),
    prevent_initial_call=True
)   
def update_output(n_clicks, num_room_value, full_sq_value, life_sq_value, kitch_sq_value,floor_value,state_value,max_floor_value, material_value, product_type_value, price_value, #ttk_km_value, metro_min_avto_value, metro_km_avto_value, metro_min_walk_value, 
metro_km_walk_value):
		X_test_dict = {
		"num_room": num_room_value,
		"full_sq": np.log(full_sq_value +1),
		"life_sq": life_sq_value,
		"floor": floor_value,
		"state": state_value,
		"max_floor": max_floor_value, 
		"material": material_value,
		"kitch_sq": kitch_sq_value,
		"product_type_num": product_type_value,
#        "ttk_km": ttk_km_value,
#        "metro_min_avto": metro_min_avto_value,
#    	 "metro_km_avto": metro_km_avto_value,
#        "metro_min_walk": metro_min_walk_value,
		"metro_km_walk": metro_km_walk_value,
		"price_doc": price_value
		}
		X_test = pd.DataFrame([X_test_dict])
		y_pred = classifier.predict(X_test)
		sub_area_1 = df[df['id']==y_pred[0][0]].sub_area_rus.values[0]
		sub_area_text = 'Месторасположение квартиры : ' + sub_area_1
		return sub_area_text, sub_area_1

@callback(
    Output('full_sq', 'value'),
    Input('full_sq', 'value'),
	Input('life_sq', 'value'),
	Input('kitch_sq', 'value'))
def update_output(full_sq_value,life_sq_value,kitch_sq_value):
	if life_sq_value+kitch_sq_value > full_sq_value:
		result = life_sq_value+kitch_sq_value
	else:
		result = full_sq_value
	return result
	
@callback(
    Output('max_floor', 'value'),
    Input('floor', 'value'),
	Input('max_floor', 'value'))
def update_output(floor_value,max_floor_value):
	if floor_value > max_floor_value:
		result = floor_value
	else:
		result = max_floor_value
	return result	

if __name__ == '__main__':
    app.run_server(debug=True)