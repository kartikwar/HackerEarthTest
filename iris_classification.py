import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

def map_varities(species, mapping_dict):
	species = mapping_dict[species]
	return species

def visualize_data(train):
	print train.describe(include='all')
	sns.barplot(x="SepalLengthCm", y="Species", data=train)
	#graph shows sepallength is a good parameter
	sns.barplot(x="SepalWidthCm", y="Species", data=train)
	# graph shows sepalwidth is a good parameter
	sns.barplot(x="PetalLengthCm", y="Species", data=train)
	# graph shows petallength is a good parameter
	sns.barplot(x="PetalWidthCm", y="Species", data=train)
	# graph shows petalwidth is a good parameter
	plt.show()

if __name__ == '__main__':
	train = pd.read_csv('Iris.csv')
	visualize_data(train)
	Iris_varieties_mapping = {'Iris-setosa' : 0  , 'Iris-versicolor' : 1 , 
	'Iris-virginica' : 2}
	train["Species"] = train["Species"].apply(map_varities, 
	args = (Iris_varieties_mapping,))