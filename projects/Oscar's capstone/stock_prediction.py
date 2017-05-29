from yahoo_finance import Share
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score
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

def save_file(stock):
	""" This function saves the current dataset"""
	d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
	'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
	df = pd.DataFrame(d)
	print df

	df.to_csv(get_file_name(stock))

def read_from_file(stock):
	#index_col = 'Date'
    dataset = pd.read_csv(get_file_name(stock), parse_dates = True, index_col = 'Date',
    	usecols = ['Date','Open','High','Low','Close','Volume', 'Adj Close'])
    #print dataset.head()

    print dataset.corr()

    y_all = dataset['Adj Close']
    X_all = dataset.drop(['Adj Close'], axis = 1)
    #X_all = dataset.drop(['Date'], axis = 1)


    #X_all['Trade Date'] = X_all.index
    #date_ordinal = X_all['Trade Date'].apply(lambda x: x.toordinal())
    
    #X_all = X_all.drop(['Trade Date'], axis = 1)
    #X_all = X_all.join(date_ordinal)
    #X_all = X_all['Trade Date'].apply(lambda x: x.toordinal())
    #print date_ordinal.head()

    print X_all.head()
    print y_all.head()

    #print X_all.head()
    # y = pd.read_csv(get_file_name(stock), index_col = 'Date', parse_dates = True, 
    #	usecols = ['Date','Adj Close'])
    #print y.head()
    return X_all, y_all

def plot(stock, df, column = 'Adj Close'):
	ax = df[column].plot(title = '{} prices'.format(stock), fontsize = 10, color='blue', linewidth=3)
	ax.set_ylabel('Price')
	ax.set_xlabel('Date')
	plt.show()

def plot_results(X_test, X_train, y_test, y_train, model, y):
	stock_date_test = pd.DataFrame()
	stock_date_train = pd.DataFrame()
	stock_date_test['Date'] = X_test.index
	stock_date_train['Date'] = X_train.index


	fig, ax = plt.subplots()
	ax.plot_date(stock_date_test, y_test, color ='r', label = 'Test data', linewidth=1)
	ax.plot_date(stock_date_train, y_train, color ='b', label = 'Train data', linewidth=1)
	#plt.plot(X_test, model.predict(y_test), color = 'b')
	#ax.set_ylabel('Price')
	#ax.set_xlabel('Date')

	
	#print "=======================X_test{}".format(X_test.shape)
	#print "+++++++++++++++++++++{}".format(y_test.shape)
	

	#stock_date_test = stock_date_test.sort_values(['Date'])
	#[y.min(), y.max()], [y.min(), y.max()]
	#print stock_date_test.head()
	#ax.plot(stock_date_test, model.predict(X_test), color='black', linewidth=1)
	#ax.plot(stock_date_test, y_test, 'k--', lw=4)
	#plt.clf()
	#plt.scatter(X_test['High'], y_test, color ='r', label = 'Test data', linewidth=1)
	#plt.scatter(X_train['High'], y_train, color ='b', label = 'Train data', linewidth=1)
	#plt.plot(X_test['High'], model.predict(X_test), color='black', linewidth=1)



	plt.legend()
	plt.axis('tight')
	#plt.title('Model {}'.format(model))
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
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	return X_train, X_test, y_train, y_test

def linear_model(X, y):
	print "********************LinearRegression Model******************"
	X_train, X_test, y_train, y_test = split_data(X, y)
	model = LinearRegression()
	model.fit(X_train, y_train)
	print '{}'.format(model.score(X_test, y_test))
	#print 'Predict {}'.format(model.predict(X_test))
	#stock_date_test = 
	#stock_price_test = y_test.drop('Date', 1)
	print "X_train {}".format(X_train.head())
	print "y_train {}".format(y_train.head())
	
	predicted = cross_val_predict(model, X, y, cv=10)
	scores = cross_val_score(model, X, y, cv=10)
	print "Scores {}". format(scores)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	fig, ax = plt.subplots()
	ax.scatter(y, predicted)
	ax.plot([y.min(), y.max()], [y_test.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()

	#print "{}".format(y_test['Adj Close'])
	#print "{}".format(stock_price_test)
	#print "{}".format(stock_date_test)
	#print "{}".format(y_test)
	plot_results(X_test, X_train, y_test, y_train, model, y)
	return model



def svr_model(X, y):
	X_train, X_test, y_train, y_test = split_data(X, y)
	#'kernel':('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
	parameters = {'kernel':['rbf'],
	 'C':[1, 10], 'gamma':[0.1]}

	#model = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10,
    #               param_grid={"C": [1e0, 1e1, 1e2, 1e3],
    #                           "gamma": np.logspace(-2, 2, 5)})
	model = SVR(C=1.0, epsilon=0.2, kernel ='rbf')

	predicted = cross_val_predict(model, X, y, cv=10)
	scores = cross_val_score(model, X, y, cv=10)
	print "Scores {}". format(scores)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	fig, ax = plt.subplots()
	ax.scatter(y, predicted)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()
	#model = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
    #               param_grid={"C": [1e0, 1e1, 1e2, 1e3],
    
    #                           "gamma": np.logspace(-2, 2, 5)})
	model.fit(X_train, y_train)
	print '{}'.format(model.score(X_test, y_test))

	plot_results(X_test, X_train, y_test, y_train, model, y)

def lasso_model(X, y):
	X_train, X_test, y_train, y_test = split_data(X, y)
	model = Lasso(alpha=0.1, max_iter=10000)
	model.fit(X_train, y_train)
	print '{}'.format(model.score(X_test, y_test))


def neighbors_model(X, y):
	X_train, X_test, y_train, y_test = split_data(X, y)
	parameters = {'weights':('uniform', 'distance'), 'n_neighbors':[2,3,5]}
	kn = KNeighborsRegressor()
	model = GridSearchCV(kn, parameters)

	predicted = cross_val_predict(model, X, y, cv=10)
	scores = cross_val_score(model, X, y, cv=10)
	print "Scores {}". format(scores)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	fig, ax = plt.subplots()
	ax.scatter(y, predicted)
	ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()



	model.fit(X_train, y_train) 
	print '{}'.format(model.score(X_test, y_test))
	#plot_results(X_test, X_train, y_test, y_train, model, y)



def run():
	#save_file('Google')
	stock = main(sys.argv[1:])
	X, y = read_from_file(stock)

	
	#linear_model(X, y)
	lasso_model(X, y)



if __name__ == '__main__':
    run()