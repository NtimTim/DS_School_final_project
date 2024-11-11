from sklearn.preprocessing import LabelEncoder
import numpy as np
import catboost as cb
import pandas as pd
import pickle

train_df = pd.read_csv('train.csv')
train_df['state'].fillna(train_df['state'].mode().iloc[0],inplace=True)
train_df['product_type_num'] = np.where(train_df['product_type'].isnull(),0 ,np.where(train_df['product_type']=='Investment',1,2))
train_df.drop(train_df.index[7457], inplace=True)

train_df['full_sq'] = np.where(train_df['full_sq']<train_df['life_sq'],train_df['life_sq'],train_df['full_sq'])
train_df.drop(train_df.index[17932], inplace=True) 
train_df['life_sq'] = np.where(train_df['life_sq']<train_df['full_sq']*0.3,train_df['full_sq']*0.5,train_df['life_sq'])
train_df['floor'] = np.where(train_df['floor']==77,7,train_df['floor'])
train_df['max_floor'] = np.where(train_df['max_floor']<train_df['floor'],train_df['floor'],train_df['max_floor'])
train_df['num_room']= np.where(train_df['num_room']>10,1,train_df['num_room'])
train_df['kitch_sq'] =  np.where(train_df['kitch_sq']>=train_df['full_sq']*0.5, train_df['full_sq']*0.3, train_df['kitch_sq'])

train_df['full_sq'] = np.log(train_df['full_sq']+1)
#numeric_data = train_df.select_dtypes([np.number])
#numeric_data_mean = numeric_data.mean()
#numeric_features = numeric_data.columns
#train_df = train_df.fillna(numeric_data_mean)[numeric_features]

lbl = LabelEncoder()
lbl.fit(list(train_df['sub_area'].values)) 
train_df['sub_area_num'] = lbl.transform(list(train_df['sub_area'].values))

for col in train_df.columns:
    if train_df[col].isnull().sum() > 0:
        mean = train_df[col].mean()
        train_df[col] = train_df[col].fillna(mean)
		
y_train = train_df['price_doc']
X_train = train_df[[
                'num_room',
				'full_sq',
				'life_sq',
				'floor',
				'state',
            	'max_floor', 
				'material',
				'kitch_sq',
				'product_type_num',
#                'ttk_km',
#                'metro_min_avto',
#                'metro_km_avto',
#                'metro_min_walk',
				'metro_km_walk',
                  'sub_area_num'
                 ]]
train_dataset = cb.Pool(X_train, np.log(y_train+1))

model = cb.CatBoostRegressor(loss_function='RMSE')
grid = {'iterations': [150, 200],
        'learning_rate': [0.03, 0.05],
        'depth': [5, 7],
        'l2_leaf_reg': [1]}

model.grid_search(grid, train_dataset)		

with open('./model.pkl','wb') as model_pkl:
	pickle.dump(model,model_pkl)
	


