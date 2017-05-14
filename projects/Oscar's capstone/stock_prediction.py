from yahoo_finance import Share

def run():
	google = Share('GOOG')
	print "Open: {}".format(google.get_open())
	print "Close: {}".format(google.get_price())


if __name__ == '__main__':
    run()