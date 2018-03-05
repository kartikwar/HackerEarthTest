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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def map_varities(species, mapping_dict):
	species = mapping_dict[species]
	return species

def save_dataframe_to_csv(dataframe, csv_file):
	dataframe.to_csv(csv_file, index=False)

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
		'AUD' : 1.29, 'NZD' : 1.38, 'EUR' : 0.81, 'SEK' : 8.26, 
		'NOK' : 7.83, 'DKK' : 6.05, 'CHF' : 1.07, 'HKD' : 0.13, 
		'SGD' : 0.76, 'MXN' : 0.053}
	return currency_conversion[ele]

def get_length(ele):
	try:
		return len(ele)
	except:
		return 0		

def preprocess_data(train, test):
	# print dataset['desc'].head()
	
	datasets = [train, test.copy()]
	preprocessed_datasets = []

	for dataset in datasets:
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
			'created_at', 'launched_at', 'deadline', 'backers_count']
		columns = [col for col in columns if col not in remove_columns]
		print columns
		dataset = dataset[columns]
		preprocessed_datasets.append(dataset)

	return preprocessed_datasets[0], preprocessed_datasets[1]	

def train_model_in_folds(counter_kf_ele, x_train, y_train, x_test, oof_train, oof_test_skf, clf):

	x_tr = x_train.iloc[counter_kf_ele['indexes'][0]]
	y_tr = y_train.iloc[counter_kf_ele['indexes'][0]]
	x_te = x_train.iloc[counter_kf_ele['indexes'][1]]
	clf.fit(x_tr, y_tr)
	oof_train[counter_kf_ele['indexes'][1]] = clf.predict(x_te)
	oof_test_skf[counter_kf_ele['counter'], :] = clf.predict(x_test)

def get_oof(clf, x_train, y_train, x_test):
	# print (type(x_train))
	ntrain = x_train.shape[0]
	# print ntrain
	ntest = x_test.shape[0]
	oof_train = np.zeros((ntrain,))
	oof_test = np.zeros((ntest,))
	SEED = 0 # for reproducibility
	NFOLDS = 10 # set folds for out-of-fold prediction
	kf = KFold(len(x_train), n_folds= NFOLDS, random_state=SEED)
	kf = list(kf)
	oof_test_skf = np.empty((NFOLDS, ntest))
	
	counter = 0
	counter_kf_list = []
	for ele in kf:
		counter_kf = {'counter' : counter, 'indexes' : ele}
		counter_kf_list.append(counter_kf)

	with poolcontext(processes=3) as pool:
		results = pool.map(partial(train_model_in_folds, 
			x_train=x_train, y_train=y_train, oof_train=oof_train, oof_test_skf=oof_test_skf, clf=clf, x_test=x_test), counter_kf_list)
		# print (oof_test_skf)
		oof_test[:] = oof_test_skf.mean(axis=0)	
		return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

def first_level_training(dataset, test):
	#visualize coorelation after preprocessing
	dataset = dataset.apply(preprocessing.LabelEncoder().fit_transform)
	test = test.apply(preprocessing.LabelEncoder().fit_transform)
	total_features = (list(dataset.columns))
	total_test_features = (list(test.columns))

	print total_features
	print total_test_features

	corr = dataset.corr(method='pearson')
	corr = corr[['final_status']]
	save_dataframe_to_csv(corr, 'output2.csv')

	y = dataset["final_status"]
	X = dataset.drop('final_status', axis=1)

	X = StandardScaler().fit_transform(X)
	test = StandardScaler().fit_transform(test)
	# print X.head()
	# X_train , X_test, y_train , y_test = train_test_split(X, y, random_state=0)

	# classifier = RandomForestClassifier(random_state=0, n_estimators=500,max_features='sqrt', n_jobs=-1).fit(X_train, y_train)
	# classifier = SVC(kernel='rbf', C=0.025, random_state=0, gamma='auto').fit(X_train, y_train)
	# classifier = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
	classifier = MLPClassifier(solver='sgd', alpha=1e-5,
		hidden_layer_sizes=(100,), random_state=1, learning_rate='adaptive', tol=1e-4).fit(X, y)
	return classifier, test

def calculate_accuracy(predicted, true):
	accuracy = accuracy_score(true, predicted)
	return accuracy

def predict_test(classifier, X_test):
	X_test_predict = classifier.predict(X_test)
	return X_test_predict 		

def second_level_training(X_train, y_train):
	gbm = xgb.XGBClassifier(n_estimators= 2000,max_depth= 4,min_child_weight= 2,
		gamma=0.9,subsample=0.8, objective='binary:logistic', 
		nthread= -1,scale_pos_weight=1).fit(X_train, y_train)
	return gbm

if __name__ == '__main__':
	dataset = pd.read_csv('train.csv')
	test_dataset = pd.read_csv('test.csv')
	visualize_data(dataset)
	dataset, test = preprocess_data(dataset, test_dataset)
	print test.head()
	# dataset = preprocess_data(dataset)
	classifier, test = first_level_training(dataset, test)
	# clf = second_level_training(X_train, y_train)
	predictions = classifier.predict(test)
	print 'predictions are ', predictions
	test_dataset['final_status'] = predictions
	save_dataframe_to_csv(test_dataset[['project_id', 'final_status']], 'submission.csv')

	# X_test_predict = predict_test(classifier, X_test)