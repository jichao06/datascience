from sklearn.linear_model import LinearRegression

data = pd.read_csv('/Users/jzhn/Documents/2014/0312 Freight Allowance Negotiation/agg_dsi_all_vendors_since_20110101.tsv', sep='\t')

lr = LinearRegression(fit_intercept=True)
fig,axes = plt.subplots(2, 3, figsize=(18,10))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
for vendor, axe in zip(['CARIZ','GERGO', 'HANMS', 'DODN9', 'HAGAC'], axes.reshape(6)):
    df = data[(data.vendor == vendor)&(data.gl==193)]
    df.received_day = pd.to_datetime(df.received_day)
    df['month'] = df.received_day.map(lambda x: x.year*100 + x.month)
    df = df.groupby('month', as_index=False)['Units'].sum()
    df['monthly'] = range(len(df))

    axe.plot(df['monthly'], df['Units'])
    ticks = range(0,40,5)
    labels = df.ix[ticks, 'month']
    axe.set_xticks(ticks)
    axe.set_xticklabels(labels.values, rotation=45)
    axe.grid()

    lr.fit(df.ix[:36, ['monthly']], df.ix[:36, 'Units'])
    axe.plot(df['monthly'], lr.predict(df[['monthly']]))

    axe.set_title('%s %.2f' % (vendor, lr.score(df.ix[:36,['monthly']], df.ix[:36,'Units'])))
    
    print vendor, lr.predict(df.ix[36:39, ['monthly']]).sum()