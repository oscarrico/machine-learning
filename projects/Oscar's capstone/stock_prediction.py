from yahoo_finance import Share
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import sys, getopt


def get_file_name(stock):
	#now = datetime.datetime.now().strftime("%Y-%m-%d")
	#return 'datasets/{}_{}.csv'.format(stock, now)
	return 'datasets/{}.csv'.format(stock)

def save_file(stock):
	""" This function saves the current dataset"""
	d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
	'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
	df = pd.DataFrame(d)
	print df

	df.to_csv(get_file_name(stock))

def read_from_file(stock):
    x = pd.read_csv(get_file_name(stock), index_col = 'Date', parse_dates = True, 
    	usecols = ['Date','Open','High','Low','Close','Volume'])
    print x.head()
    y = pd.read_csv(get_file_name(stock), index_col = 'Date', parse_dates = True, 
    	usecols = ['Date','Adj Close'])
    print y.head()
    return x , y

def plot(stock, df, column = 'Adj Close'):
	ax = df[column].plot(title = '{} prices'.format(stock), fontsize = 10)
	ax.set_ylabel('Price')
	ax.set_xlabel('Date')
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
   if stock == '':
   	print 'stock_prediction.py -s <stock_symbol>'
   	sys.exit()
   else:
   	print "Symbol {}".format(stock)
   	return stock

def run():
	#save_file('Google')
	stock = main(sys.argv[1:])
	x, y = read_from_file(stock)
	#plot(stock, y)



	#start = datetime.datetime(2017, 5, 10)
	#end = datetime.datetime(2017, 5, 14)
	#df = web.DataReader("GOOGL", 'yahoo', start, end)
	#print df.head()
	#print df['Adj Close']
	#df['Adj Close'].plot()
	#plt.show()
	#google = Share('YHOO')
	#print google.get_historical('2014-04-25', '2014-04-29')
	#print "Open: {}".format(google.get_open())
	#print "Close: {}".format(google.get_price())


if __name__ == '__main__':
    run()