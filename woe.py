# для непрерывных переменных

def Woe_IV_cont(df, features, target):
    
    aux = features + [target] 
    
    df = df[aux].copy()
    df_woe_iv = pd.DataFrame({},index=[])   # создаётся пустой df
    
    # количество target = 1
    _t1 = sum(df[target])
    # количество target = 0
    _t0 =  len(df[target]) - _t1
    
    # разбиваем на квантили непрерывные переменные
    _quantile = df.iloc[:, :-1].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9], axis = 0)
    
    
    for _column in _quantile.columns:
   
        # удаление дублей квантилей 
        list_aux = _quantile[[_column]].drop_duplicates().to_numpy()
    
        _tiv = 0
        
        print(list_aux)
        
        for q in range(len(list_aux)):
            
            
            if q>0:
                location = df[(df.loc[:,_column] > float(list_aux[q-1])) & (df.loc[:,_column] <= float(list_aux[q]))].index   # список с индексами
                limit = str(list_aux[q-1]) + ' a ' + str(list_aux[q])    # строка для отображения границ квантиля

                print(location)
            else:
                location = df[(df.loc[:,_column] <= float(list_aux[q]))].index
                limit = '<=' + str(list_aux[q])

                print(location)
                
            _many = len(location)  
            
            # Target = 1 дефолт
            _1 = sum(df.loc[location,target])
            _p1 = _1/_t1
            
            # Target = 0 недефолт
            _0 = _many - _1
            _p0 = _0/_t0
            
            # отношение не дефолтных к дефолтам
            if _p1 == 0 or _p0 == 0:
                _Distr = 1
            else:
                _Distr = _p0/_p1
            
            # Weight of evidence
            _woe = np.log(_Distr)
            
            # IV
            _iv = round(_woe*(_p0-_p1),2)
            
            # IV - total
            _tiv = _tiv+_iv
                    
            
            dframe = pd.DataFrame({'variable': _column , 'limit':limit , '0': _p0 , '1': _p1, 'Distr':_Distr, 'WoE': _woe , 'IV':  _iv}
                                  , index = [ _column])  
            
            df_woe_iv = pd.concat([df_woe_iv, dframe], ignore_index=True)
            
        dframe = pd.DataFrame({'variable': _column ,'limit': ' ' , '0': 1 , '1': 1, 'Distr': 1 , 'WoE': 0 , 'IV':  _tiv}
                                  , index =[ _column])
         
        df_woe_iv = pd.concat([df_woe_iv, dframe], ignore_index=True)
            
    return df_woe_iv

# для дискретных
def Woe_IV_Dis(df, features, target):
    aux = features + [target] 
    df = df[aux].copy()
    df_woe_iv = pd.DataFrame({},index=[])
    for feature in features:
        df_woe_iv_aux = pd.crosstab(df[feature], df[target], normalize='columns')\
        .assign(Distr=lambda dfx: dfx[0] / dfx[1])\
        .assign(WoE=lambda i: np.log(i[0] / i[1]))\
        .assign(IV=lambda i: (i['WoE']*(i[0]-i[1])))\
        .assign(IV_total=lambda i: np.sum(i['IV']))
        
        df_woe_iv = pd.concat([df_woe_iv, df_woe_iv_aux])#, levels=['fico_flag','recent_inquiries_gr','Nb_active_mortgages','micro_total_gr','active_total_gr','Total_overdue_amount_gr','channel_value','insurance','max_day','end_order_value''avg_amountorders_gr','share_early_payment'])

    return df_woe_iv  


#обхединяем

def Woe_IV(df, features_dis, features_cont, target):
    p=[(features_dis[i]+' ')*rp[features_dis[i]].nunique() for i in range(len(features_dis))]
    p=[p[i].split(' ') for i in range(len(p))]
    p=[list(filter(None, p[i])) for i in range(len(p))]
    df_dis =  Woe_IV_Dis(df, features_dis, target)
    df_cont =  Woe_IV_cont(df, features_cont, target)
    
    df_dis.reset_index(inplace=True)
    df_dis = df_dis.rename(columns = {'index':'variable',0: '0', 1: '1'})
    df_dis.insert(loc = 1, column = 'limit', value = ' ')

  #  for i in range(len(df_dis['variable'])):
  #      print(i)
  #      if df_dis.at[i, 'variable']==0:
  #          df_dis.at[i, 'limit']=feature_dis[k]
  #          k=k+1

    df_cont['IV_total'] = ' '
    k=0
    for i in range(len(p)):
        for j in range(len(p[i])):
            df_dis.at[k, 'limit']=p[i][j]
            k+=1
    df_woe_iv = pd.concat([df_dis, df_cont])
    
    return df_woe_iv


features_dis=[]
features_cont=[]
target='90_6mob'
feature=rp.loc[:, rp.columns !=target]
for i in feature:
    if rp[i].nunique()>=10:
        features_cont.append(i)
    else:
        features_dis.append(i)


res=Woe_IV(rp, features_dis, features_cont, target)



