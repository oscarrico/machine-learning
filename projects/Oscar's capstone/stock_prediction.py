from yahoo_finance import Share
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.dummy import DummyRegressor
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import sys, getopt
import numpy as np

STOCKS = ['GOOG', 'AAPL', 'AMZN', 'MSFT']

def get_file_name(stock):
    #Method to read file from datasets folder
    return 'datasets/{}.csv'.format(stock)

def read_from_file(stock):
	#index_col = 'Date'
    dataset = pd.read_csv(get_file_name(stock), parse_dates = True, index_col = 'Date',
    	usecols = ['Date','Open','High','Low','Close','Volume', 'Adj Close'])
    #print dataset.head()
    print dataset.corr()
    y_all = dataset['Adj Close']
    X_all = dataset.drop(['Adj Close'], axis = 1)
    print X_all.head()
    print y_all.head()
    return X_all, y_all

def plot(stock, df, column = 'Adj Close'):
	ax = df[column].plot(title = '{} prices'.format(stock), fontsize = 10, color='blue', linewidth=3)
	ax.set_ylabel('Price')
	ax.set_xlabel('Date')
	plt.show()

def plot_split_results(X_test, X_train, y_test, y_train):
	stock_date_test = pd.DataFrame()
	stock_date_train = pd.DataFrame()
	stock_date_test['Date'] = X_test.index
	stock_date_train['Date'] = X_train.index


	fig, ax = plt.subplots()
	ax.plot_date(stock_date_test, y_test, color ='r', label = 'Test data', linewidth=1)
	ax.plot_date(stock_date_train, y_train, color ='b', label = 'Train data', linewidth=1)
	ax.set_xlabel('Date')
	ax.set_ylabel('Price')

	plt.legend()
	plt.title('Splitted dataset')
	plt.show()

def plot_tunned_model(name, model, X, y):
	predicted = cross_val_predict(model, X, y, cv=10)
	scores = cross_val_score(model, X, y, cv=10)
	print "Scores {}". format(scores)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	fig, ax = plt.subplots()
	ax.scatter(y, predicted, label = 'Predicted Price', color ='r')
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, label = 'Model')
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.legend()
	plt.title('{} trainned model.'.format(name))
	plt.show()

def main(argv):
   stock = ''
   try:
      opts, args = getopt.getopt(argv,"hs:")
   except getopt.GetoptError:
      print 'stock_prediction.py -s <stock_symbol>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'stock_prediction.py -s <stock_symbol>'
         sys.exit()
      elif opt == '-s':
         stock = arg
         if stock not in STOCKS:
         	print "Invalid stock symbol {}".format(stock)
         	print "Valid stock symbols are: {}".format(STOCKS) 
         	sys.exit()
   if stock == '':
   	print 'stock_prediction.py -s <stock_symbol>'
   	sys.exit()
   else:
   	print "Symbol {}".format(stock)
   	return stock

def split_data(X, y):
	print "********************Splitting the Data******************"
	print X.shape
	print y.shape
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
	return X_train, X_test, y_train, y_test

def benchmark(X_train, X_test, y_train, y_test):
	print "********************DummyRegressor Model******************"
	model = DummyRegressor()
	model.fit(X_train, y_train)
	print '{}'.format(model.score(X_test, y_test))
	return model

def linear_model(X_train, X_test, y_train, y_test):
	print "********************LinearRegression Model******************"
	model = LinearRegression()
	model.fit(X_train, y_train)
	print '{}'.format(model.score(X_test, y_test))
	return model

def linear_model_tunned(X_train, X_test, y_train, y_test, X, y):
	print "********************LinearRegression Tunned Model******************"
	parameters = {'fit_intercept':[True, False], 'normalize':[True, False]}
	model = LinearRegression()
	model = GridSearchCV(model, parameters)
	model.fit(X_train, y_train)
	print '{}'.format(model.score(X_test, y_test))
	plot_tunned_model('LinearRegression', model, X, y)



def svr_model(X_train, X_test, y_train, y_test, X, y):
	print "********************Epsilon-Support Vector Regression Model******************"
	#'kernel':('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
	#parameters = {'kernel':['rbf'], 'C':[1.0, 10], 'gamma':[0.1, 'auto']}

	model = SVR()
	#model = GridSearchCV(model, parameters)
	model.fit(X_train, y_train)
	print '{}'.format(model.score(X_test, y_test))
	#model = model.best_estimator_
	#print "Parameters are {} for the optimal model.".format(model.get_params())
	plot_tunned_model('SVR', model, X, y)

def lasso_model(X_train, X_test, y_train, y_test):
	print "********************Lasso Model******************"
	model = Lasso(random_state=42)
	model.fit(X_train, y_train)
	print '{}'.format(model.score(X_test, y_test))

def lasso_model_tunned(X_train, X_test, y_train, y_test, X, y):
	print "********************Lasso Model Tunned Model******************"
	parameters = {'alpha': [0.1, 0.5, 1.0], 'max_iter':(1000,10000, 150000), 'fit_intercept': (True, False), 'selection': ('random', 'cyclic')}
	model = Lasso(random_state=42)
	model = GridSearchCV(model, parameters)
	model = model.fit(X_train, y_train)
	print '{}'.format(model.score(X_test, y_test))
	plot_tunned_model('Lasso', model, X, y)


def neighbors_model(X_train, X_test, y_train, y_test, X, y):
	print "********************KNeighborsRegressor Model******************"

	#parameters = {'weights':('uniform', 'distance'), 'n_neighbors':[2,3,5]}
	model = KNeighborsRegressor()
	#model = GridSearchCV(kn, parameters)
	model.fit(X_train, y_train)
	print '{}'.format(model.score(X_test, y_test))
	#model = model.best_estimator_
	#print "Parameters are {} for the optimal model.".format(model.get_params())
	plot_tunned_model('KNeighborsRegressor', model, X, y)


def run():
	#save_file('Google')
	stock = main(sys.argv[1:])
	X, y = read_from_file(stock)
	X_train, X_test, y_train, y_test = split_data(X, y)
	plot_split_results(X_train, X_test, y_train, y_test)

	benchmark(X_train, X_test, y_train, y_test)
	#svr_model(X_train, X_test, y_train, y_test, X, y)
	#neighbors_model(X_train, X_test, y_train, y_test, X, y)
	#linear_model(X_train, X_test, y_train, y_test)
	#lasso_model(X_train, X_test, y_train, y_test)
	#linear_model_tunned(X_train, X_test, y_train, y_test, X, y)
	lasso_model_tunned(X_train, X_test, y_train, y_test, X, y)



if __name__ == '__main__':
    run()