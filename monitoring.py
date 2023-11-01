import traceback
df_gini=df.loc[(df['DESCRIPTION'].isin(['Типовая'])) & (df['REGUL_TECHNIQUE_DECISION'].isin(['Одобрить','Отказать']))] 
MODEL='PD_TOTAL'
INDICATORS=['MOB1P_DPD_RESTR_BANKR','MOB30P_DPD_RESTR_BANKR','MOB90P_DPD_RESTR_BANKR']

gini_farme=pd.DataFrame()
m=df_gini.MM_YYY.unique()
mobs=df_gini.MOB.unique()

for MM_YYY in m:
    for mob in mobs:
        for INDICATOR in INDICATORS:
            if ((df_gini.loc[(df_gini['MM_YYY']==MM_YYY)&(df_gini['MOB']==mob)][INDICATOR]).sum()!=0)&((df_gini.loc[(df_gini['MM_YYY']==MM_YYY)&(df_gini['MOB']==mob)][INDICATOR]).count()>1): 

                gini=-1+2*roc_auc_score(df_gini.loc[(df_gini['MM_YYY']==MM_YYY) \
                                                                              & (df_gini['MOB']==mob)][INDICATOR],\
                                            df_gini.loc[(df_gini['MM_YYY']==MM_YYY) \
                                                                       & (df_gini['MOB']==mob)][MODEL])

                gini_farme['MM_YYY']=pd.to_datetime(MM_YYY)
                gini_farme['MODEL']=[MODEL]
                gini_farme['MOB']=mob
                gini_farme['INDICATOR']=INDICATOR
                gini_farme['GINI']=gini
                GINI=GINI.append(gini_farme)
import builtins as py_builtin


def psi(month_year_app, score_initial, score_new, num_bins=10, mode = 'quantile'):
    eps = 1e-4
    # Sort the data
    score_initial.sort()
    score_new.sort()
    
    
    min_val = py_builtin.min(py_builtin.min(score_initial), py_builtin.min(score_new))
    max_val = py_builtin.max(py_builtin.max(score_initial), py_builtin.max(score_new))
    
    if mode == 'fixed':
        bins = [min_val + (max_val - min_val)*(i)/num_bins for i in range(num_bins+1)]
    elif mode == 'quantile':
        #bins = pd.qcut(score_initial, q = num_bins, retbins = True).values.add_categories('missing')[1] # Create the quantiles based on the initial population
        bins = pd.qcut(score_initial, q = num_bins, retbins = True, duplicates='drop')[1] # Create the quantiles based on the initial population
        np.append(bins,-999999999)
    else:
        raise ValueError({mode})
    bins[0] = min_val - eps # Correct the lower boundary
    bins[-1] = max_val + eps # Correct the higher boundary
    
    
   
    bins_initial = pd.cut(score_initial, bins = bins)
    df_initial = pd.DataFrame({'initial': score_initial, 'bin': bins_initial})
    grp_initial = df_initial.groupby('bin').count()
    grp_initial['percent_initial'] = grp_initial['initial'] / sum(grp_initial['initial'])

    
    
    bins_new = pd.cut(score_new, bins = bins)
    df_new = pd.DataFrame({'new': score_new, 'bin': bins_new})
    grp_new = df_new.groupby('bin').count()
    grp_new['percent_new'] = grp_new['new'] / sum(grp_new['new'])
    
    
    #psi_df = grp_initial.join(grp_new, on = "bin", how = "inner")
    psi_df=grp_initial.join(grp_new).fillna(0)
    
    psi_df['percent_initial'] = psi_df['percent_initial'].apply(lambda x: eps if x == 0 else x)
    psi_df['percent_new'] = psi_df['percent_new'].apply(lambda x: eps if x == 0 else x)
    psi_df['month_year_app']=month_year_app
    psi_df['psi'] = (psi_df['percent_initial'] - psi_df['percent_new']) * np.log(psi_df['percent_initial'] / psi_df['percent_new'])
    
    return psi_df
#['psi'].values
psi_df=psi('2023-03', dev['PD'].values, data_cut_cl.loc[(data_cut_cl['month_year_app']=='2023-03') & (data_cut_cl['DESCRIPTION'].isin(['Типовая'])) & (data_cut_cl['PD_NA'] != 1)]['PD_OFFLINE'].values, mode = 'quantile')
