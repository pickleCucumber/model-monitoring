def get_psi_from_dev_val(df_dev, df_val, column_name, num_bins):
    cats, bins=pd.qcut(df_dev[column_name], q=num_bins, duplicates='drop', retbins=True)
    df_dev=df_dev.assign(labels=cats)
    res_dev=df_dev.groupby('labels', as_index=False)[column_name].count()
    
    bins[0], bins[-1]=-np.inf, +np.inf
    cats_val=pd.cut(df_val[column_name], bins)
    
    df_val=df_val.assign(labels=cats_val)
    res_val=df_val.groupby('labels', as_index=False)[column_name].count()

    psi_count=res_dev.merge(res_val, left_index=True, right_index=True, suffixes=('_val', '_val'))
    
    psi_count_short=psi_count.iloc[:,[1,3]]
    psi_count_normed=psi_count_short.div(psi_count_short.sum(axis=0), axis=1)
    #print('psi_count',psi_count)
    psi_count_normed['psi']=(psi_count_normed.iloc[:,1]-psi_count_normed.iloc[:,0])*np.log(psi_count_normed.iloc[:,1]/psi_count_normed.iloc[:,0])
    return psi_count_normed, psi_count_normed['psi'].sum()

def get_test_result_psi(df_dev, df_val, column_name, num_bins):
    p=pd.DataFrame()
    psi=[]
    for i in column_name:
        p, psi_index=get_psi_from_dev_val(df_dev, df_val, i, num_bins)
        psi.append(psi_index)
    s=pd.Series(psi, index=column_name)
    df_psi=s.to_frame(name='Значение теста PSI')
    df_psi['test_res']=np.where(df_psi['Значение теста PSI']<0.1, 'green', np.where(0.2<=df_psi['Значение теста PSI'], 'red', 'yellow'))
    return df_psi
  -------------------------------------------------------------------------------------------------------------------------------------------
def psi(i, score_initial, score_new, num_bins, mode):
    
    eps = 1e-4
    
    score_initial.sort_values(ascending=True)
    score_new.sort_values(ascending=True)
    
    min_val = min(min(score_initial), min(score_new))
    max_val = max(max(score_initial), max(score_new))
    if mode == 'fixed':
        bins = [min_val + (max_val - min_val)*(i)/num_bins for i in range(num_bins+1)]
    elif mode == 'quantile':
        bins = pd.qcut(score_initial, q = num_bins, retbins = True)[1] # Create the quantiles based on the initial population
    else:
        raise ValueError(f"Mode \'{mode}\' not recognized. Your options are \'fixed\' and \'quantile\'")
    bins[0] = min_val - eps # Correct the lower boundary
    bins[-1] = max_val + eps # Correct the higher boundary
        
  
    bins_initial = pd.cut(score_initial, bins = bins, labels = range(1,num_bins+1))
    df_initial = pd.DataFrame({'initial': score_initial, 'bin': bins_initial})
    grp_initial = df_initial.groupby('bin').count()
    grp_initial['percent_initial'] = grp_initial['initial'] / sum(grp_initial['initial'])
    
    bins_new = pd.cut(score_new, bins = bins, labels = range(1,num_bins+1))
    df_new = pd.DataFrame({'new': score_new, 'bin': bins_new})
    grp_new = df_new.groupby('bin').count()
    grp_new['percent_new'] = grp_new['new'] / sum(grp_new['new'])
    
    psi_df = grp_initial.join(grp_new, on = "bin", how = "inner")
    # Add a small value for when the percent is zero
    psi_df['percent_initial'] = psi_df['percent_initial'].apply(lambda x: eps if x == 0 else x)
    psi_df['percent_new'] = psi_df['percent_new'].apply(lambda x: eps if x == 0 else x)
    psi_df['feature']=i
    # Calculate the psi
    psi_df['psi'] = (psi_df['percent_initial'] - psi_df['percent_new']) * np.log(psi_df['percent_initial'] / psi_df['percent_new'])

    return psi_df['psi'].sum()

  -------------------------------------------------------------------------------------------------------------------------------------------

def calculate_psi(expected, actual, buckets, buckettype='bins', axis=0):

    def psi(expected_array, actual_array, buckets):
        
        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_fractions = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_fractions = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)
        #    psi_df['psi'] = (psi_df['percent_initial'] - psi_df['percent_new']) * np.log(psi_df['percent_initial'] / psi_df['percent_new'])


        psi_value = sum(sub_psi(expected_fractions[i], actual_fractions[i]) for i in range(0, len(expected_fractions)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[1 - axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)
