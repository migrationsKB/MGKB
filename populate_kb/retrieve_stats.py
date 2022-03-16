import tradingeconomics as te

te.login("emh16qa5qy7p8r4:f5i1mgxb4111z1w")

data = te.getHistoricalData(country='United Kingdom', indicator='Long Term Unemployment Rate', initDate='2013-01-01')

print(data)

