import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#ignore warnings
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

def map_varities(species, mapping_dict):
	species = mapping_dict[species]
	return species

def save_dataframe_to_csv(dataframe, csv_file):
	dataframe.to_csv(csv_file)

def determine_best_params_random_forest(X_train, y_train):
	grid_values = {'n_estimators' : [125, 625]}
	clf = RandomForestClassifier(random_state = 0)
	grid_clf_accuracy = GridSearchCV(clf, param_grid=grid_values, 
		n_jobs=-1, scoring='accuracy')
	grid_clf_accuracy.fit(X_train, y_train)
	best_params =grid_clf_accuracy.best_params_
	return best_params

def visualize_data(df):
	# print list(df['currency'].unique())
	# df =  df[['backers_count']]
	df = df.copy()
	df = df.apply(preprocessing.LabelEncoder().fit_transform)
	corr = df.corr(method='pearson')
	corr = corr[['final_status']]
	# print corr
	save_dataframe_to_csv(corr, 'output.csv')	
	# print df.describe()

def convert_currency_to_us(ele):
	currency_conversion = {'USD' : 1.0, 'GBP' : 0.73, 'CAD' : 1.29, 
		'AUD' : 1.29, 'NZD' : 1.38, 'EUR' : 0.81, 'SEK' : 8.26, 'NOK' : 7.83, 'DKK' : 6.05}
	return currency_conversion[ele]

def get_length(ele):
	try:
		return len(ele)
	except:
		return 0		

def preprocess_data(dataset):
	# print dataset['desc'].head()
	# convert all goal prices to USD for comparision
	dataset['currency'] = dataset['currency'].apply(convert_currency_to_us)
	dataset['price'] = dataset['goal'] * dataset['currency']
	
	#get the difference between important dates
	dataset['dead_create'] = dataset['deadline'] - dataset['created_at']
	dataset['dead_launch'] = dataset['deadline'] - dataset['launched_at']
	dataset['launch_create'] = dataset['launched_at'] - dataset['created_at']

	#get the length of description
	desc_length = dataset['desc'].apply(get_length)
	name_length = dataset['name'].apply(get_length)
	# dataset['keywords_length'] = dataset['keywords'].apply(get_length)
	dataset['length'] = desc_length + name_length
	# dataset['keywords_length'] = dataset.apply(get_length)

	# remove unnecessary features
	columns = list(dataset.columns)
	remove_columns = ['project_id', 'name', 'desc', 'state_changed_at', 'currency', 'goal', 
		'created_at', 'launched_at', 'deadline']
	columns = [col for col in columns if col not in remove_columns]
	print columns
	dataset = dataset[columns]
	return dataset	

def train_data(dataset):
	#visualize coorelation after preprocessing
	dataset = dataset.apply(preprocessing.LabelEncoder().fit_transform)
	# corr = dataset.corr(method='pearson')
	# corr = corr[['final_status']]
	# save_dataframe_to_csv(corr, 'output2.csv')

	y = dataset["final_status"]
	X = dataset.drop('final_status', axis=1)

	X = StandardScaler().fit_transform(X)

	# print X.head()
	X_train , X_test, y_train , y_test = train_test_split(X, y, random_state=0)

	#determine best params for random forest
	# best_params = determine_best_params_random_forest(X_train, y_train)
	# print 'best params for random forest are ', best_params

	rcf = RandomForestClassifier(warm_start=True,  n_jobs=-1 , verbose=0,
		min_samples_leaf=2, n_estimators=500, max_features='sqrt',
		max_depth=6, random_state = 0).fit(X_train, y_train)

	return rcf, X_train, X_test, y_train, y_test

def calculate_accuracy(predicted, true):
	accuracy = accuracy_score(true, predicted)
	return accuracy

def predict_test(classifier, X_test):
	X_test_predict = classifier.predict(X_test)
	return X_test_predict 		

if __name__ == '__main__':
	dataset = pd.read_csv('train.csv')
	visualize_data(dataset)
	dataset = preprocess_data(dataset)
	# print dataset.head()
	# dataset = preprocess_data(dataset)
	classifier, X_train, X_test, y_train, y_test = train_data(dataset)
	X_test_predict = predict_test(classifier, X_test)
	test_accuracy = calculate_accuracy(X_test_predict, y_test)
	print test_accuracy