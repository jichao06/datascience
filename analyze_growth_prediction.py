import pandas as pd
import numpy as np
import statsmodels.api as sm

def aggregate_dsi():
    data = pd.read_csv('/Users/jzhn/Documents/2014/0312 Freight Allowance Negotiation/agg_dsi_all_vendors_since_20110101.tsv', sep='\t')
    data.received_day = pd.datetools.to_datetime(data.received_day)
    data['month'] = data.received_day.map(lambda x: x.year*100+x.month)
    data = data.groupby(['vendor','gl','month'], as_index=False)['Units'].sum()
    return data

def pick_top_n_vendor_gls(agg, n):
    totals = agg.groupby(['vendor','gl'], as_index=False)['Units'].sum()
    totals.sort_index(by='Units', inplace=True, ascending=False)
    top_n = totals.iloc[:n]
    top_n = pd.merge(agg, top_n[['vendor','gl']], on = ['vendor','gl'], how='inner')
    top_n = top_n.set_index(['vendor','gl', 'month']).unstack(['vendor','gl'])
    top_n = top_n.fillna(0)
    top_n = top_n.stack(['vendor','gl']).reset_index()   
    return top_n 

def analyze_top_n(tops):
    groups = tops.groupby(['vendor','gl'])
    #result = pd.DataFrame([analyze(vendor, gl, chunk) for (vendor,gl), chunk in groups])
    result = pd.concat([analyze2(vendor, gl, chunk) for (vendor,gl), chunk in groups])
    return result

def analyze(vendor, gl, data):
    data = data.iloc[:-1].copy()
    data['sequence'] = range(len(data))
    data['const'] = 1
    
    training = data.iloc[:35]
    training.ix[training['Units']==0, 'Units']=1
    ols = sm.OLS(training['Units'], training[['const','sequence']])
    model_ols = ols.fit()
    data['Units_ols'] = model_ols.predict(data[['const','sequence']])
    
    pls = sm.WLS(training['Units'], training[['const', 'sequence']], weights = np.square(1/training['Units']))
    model_pls = pls.fit()
    data['Units_pls'] = model_pls.predict(data[['const','sequence']])
    
    wls_sequence = sm.WLS(training['Units'], training[['const', 'sequence']], weights = training['sequence'])
    model_wls_sequence = wls_sequence.fit()
    data['Units_wls_sequence'] = model_wls_sequence.predict(data[['const','sequence']])
    
    wls_mixed = sm.WLS(training['Units'], training[['const', 'sequence']], weights = training['sequence'] /training['Units'])
    model_wls_mixed = wls_mixed.fit()
    data['Units_wls_mixed'] = model_wls_mixed.predict(data[['const','sequence']])
    
    sums = data.iloc[36:][['month','Units', 'Units_ols', 'Units_pls', 'Units_wls_sequence', 'Units_wls_mixed']]
    sums['vendor'] = vendor
    sums['gl'] = gl
    return sums

def analyze2(vendor, gl, data):
    data = data.iloc[:-1].copy()
    data['sequence'] = range(len(data))
    data['const'] = 1
    
    training = data.iloc[:35]
    training.ix[training['Units']==0, 'Units']=1
    prediction = data.iloc[36:]
    
    ols = sm.OLS(training['Units'], training[['const','sequence']])
    model_ols = ols.fit()
    prediction['Units_ols'] = data.iloc[24:27]['Units'].values + 12 * model_ols.params[1]
    '''
    pls = sm.WLS(training['Units'], training[['const', 'sequence']], weights = np.square(1/training['Units']))
    model_pls = pls.fit()
    data['Units_pls'] = model_pls.predict(data[['const','sequence']])
    
    wls_sequence = sm.WLS(training['Units'], training[['const', 'sequence']], weights = training['sequence'])
    model_wls_sequence = wls_sequence.fit()
    data['Units_wls_sequence'] = model_wls_sequence.predict(data[['const','sequence']])
    
    wls_mixed = sm.WLS(training['Units'], training[['const', 'sequence']], weights = training['sequence'] /training['Units'])
    model_wls_mixed = wls_mixed.fit()
    data['Units_wls_mixed'] = model_wls_mixed.predict(data[['const','sequence']])
    '''
    sums = prediction
    sums['vendor'] = vendor
    sums['gl'] = gl
    return sums

def analyze3(vendor, gl, data):
    data = data.iloc[:-1].copy()
    data['sequence'] = range(len(data))
    data['const'] = 1
    
    regressors = ['sequence']
    for i in range(1, 13):
        column = 'month_' + str(i)
        data[column] = (data['month'] % 100 == i).astype(np.int)
        regressors.append(column)
        
        
    training = data.iloc[:27].copy()
    training.ix[training['Units']==0, 'Units']=1
    
    ols = sm.WLS(training['Units'], training[regressors], weights = np.square(training['sequence']))
    model = ols.fit()
    print model.summary()
    data['Units_ols'] = model.predict(data[regressors])
    return data
    
def analyze4(vendor, gl, data):
    data = data.iloc[:-1].copy()
    data['sequence'] = range(len(data))
    data['sequence2'] = np.square(data['sequence'])
    data['const'] = 1
    
    regressors = ['sequence', 'sequence2']
    for i in range(1, 13):
        column = 'month_' + str(i)
        data[column] = (data['month'] % 100 == i).astype(np.int)
        regressors.append(column)
        
        
    training = data.iloc[:27].copy()
    training.ix[training['Units']==0, 'Units']=1
    
    ols = sm.WLS(training['Units'], training[regressors])
    model = ols.fit()
    print model.summary()
    data['Units_ols'] = model.predict(data[regressors])
    return data
    
def main(n):
    agg = pd.read_csv('/Users/jzhn/Documents/2014/0312 Freight Allowance Negotiation/monthly_dsi.csv')
    top_n = pick_top_n_vendor_gls(agg, n)
    result = analyze_top_n(top_n)
    return result 

if __name__=='__main__':
    agg = pd.read_csv('/Users/jzhn/Documents/2014/0312 Freight Allowance Negotiation/monthly_dsi.csv')
    top_30 = pick_top_n_vendor_gls(agg, 10)
    result = analyze_top_n(top_30)
    print result