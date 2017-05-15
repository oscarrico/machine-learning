#from yahoo_finance import Share
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt


def run():
	start = datetime.datetime(2017, 5, 10)
	end = datetime.datetime(2017, 5, 14)
	df = web.DataReader("GOOG", 'yahoo', start, end)
	print df.head()
	print df['Adj Close']
	df['Adj Close'].plot()
	plt.show()
	#google = Share('GOOG')
	#print google.get_historical('2017-05-10', '2017-05-14')
	#print "Open: {}".format(google.get_open())
	#print "Close: {}".format(google.get_price())


if __name__ == '__main__':
    run()