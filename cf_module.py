import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

#The all supported fuctions for section4 collaborative filtering
def data_generator_new(data, prop, seed=None):

    if seed is not None:
        np.random.seed(seed)
    df = data.copy()
    sample = []  
    
    size = int(data.shape[0]*prop)
    index = data.index
    column = data.columns
    
    for i in range(data.shape[1]):
        row = np.random.choice(index, size, replace = False)
        df.loc[row, column[i]] = np.nan
        coordinate = [(r, column[i]) for r in row] 
        sample+=coordinate
    
    output = [df, sample]
    return output

def similarity_cosine_item(data_actual, data_na, location):
    data, result_temp = Baseline1(data_actual, data_na, location)
    similarities = cosine_similarity(data.transpose())
    sim = pd.DataFrame(data=similarities, index = data.columns,
                                          columns = data.columns)
    return sim

def similarity_cosine_user(data_actual, data_na, location):
    data, result_temp = Baseline1(data_actual, data_na, location)
    similarities = cosine_similarity(data)
    sim = pd.DataFrame(data=similarities, index = data.index,
                                          columns = data.index)
    return sim

def similarity_cosine_pca_item(data_actual, data_na, location, pc=50):
    data, result_temp = Baseline1(data_actual, data_na, location)
    data = data.transpose()
    pca = PCA(pc)
    pca.fit(data)
    pc_scores = pca.transform(data)
    similarities = cosine_similarity(pc_scores)
    sim = pd.DataFrame(data=similarities, index = data.index,
                                          columns = data.index)
    return sim

def similarity_cosine_pca_user(data_actual, data_na, location, pc=30):
    data, result_temp = Baseline1(data_actual, data_na, location)
    pca = PCA(pc)
    pca.fit(data)
    pc_scores = pca.transform(data)
    similarities = cosine_similarity(pc_scores)
    sim = pd.DataFrame(data=similarities, index = data.index,
                                          columns = data.index)
    return sim


def Find_similar(item, sim, K, NAs):
    rate = sim.loc[item,:]
    klargest = rate.drop(NAs)
    klargest = klargest.nlargest(K)
    name = klargest.index
    return name
    


#item-based
def UserItem_calculation1(data_original, location, sim, k):
    data = data_original.copy()
    location = np.array(location) 
    users = np.unique(location[:, 0]) 
    for i in range(len(users)):
        user_index = users[i] 
        data_i = data.loc[user_index] 
        items_na = data_i[data_i.isnull()].index.values 
        for na in items_na:
            k_sim = Find_similar(na, sim, K = k, NAs = items_na)
            R_uN = data_i.loc[k_sim]
            S_iN = sim.loc[na, k_sim]
            if(sum(S_iN)!=0):
                P_Ui = sum(S_iN*R_uN)/sum(S_iN)
            else:
                P_Ui = 0
            prediction = P_Ui>0.5
            prediction = int(prediction)
            data.loc[user_index, na] = prediction
    return data

#user-based
def UserItem_calculation2(data_original, location, sim, k):
    data = data_original.copy()
    location = np.array(location)
    items = np.unique(location[:, 1]) 
    for j in range(len(items)):
        items_index = items[j] 
        data_j = data.loc[:, int(items_index)] 
        user_na = data_j[data_j.isnull()].index.values 
        for na in user_na:
            k_sim = Find_similar(na, sim, K = k, NAs = user_na)
            R_uN = data_j.loc[k_sim]
            S_iN = sim.loc[na, k_sim]
            if(sum(S_iN)!=0):
                P_Ui = sum(S_iN*R_uN)/sum(S_iN)
            else:
                P_Ui = 0
            prediction = P_Ui>0.5
            prediction = int(prediction)
            data.loc[na, int(items_index)] = prediction
    return data


def Metrics(data_actual, data_pred, location):
    actual = []
    pred = []
    for i in range(len(location)):
        row = location[i][0]
        col = location[i][1]
        actual.append(data_actual.loc[row, col])
        pred.append(data_pred.loc[row, col])
        
    cm_test = confusion_matrix(actual, pred)
    
    TP = cm_test[1, 1]
    TN = cm_test[0, 0]
    FP = cm_test[0, 1]
    FN = cm_test[1, 0]
    Accuracy = (TP + TN)/(TP + FN + TN + FP)
    Sensitivity = TP/(TP + FN)
    Precision = TP/(TP + FP)
    return([Accuracy, Sensitivity, Precision])


def Baseline1(data_actual, data_na, location):
    # row mode
    actual = []
    pred = []
    data = data_na.copy()
    for i in range(len(location)):
        row = location[i][0]
        col = location[i][1]
        if(sum(data.loc[row,:].isnull()) != len(data.loc[row,:])):
            temp = data.loc[row,:].mode(dropna=True)
            if(len(temp) > 1):
                predict = np.random.randint(2)
            else:
                predict = int(data.loc[row,:].mode(dropna=True))
        else:
            predict = 0
        data.loc[row, col] = predict
        actual.append(data_actual.loc[row, col])
        pred.append(predict)
    cm_test = confusion_matrix(actual, pred)
    TP = cm_test[1, 1]
    TN = cm_test[0, 0]
    FP = cm_test[0, 1]
    FN = cm_test[1, 0]
    Accuracy = (TP + TN)/(TP + FN + TN + FP)
    Sensitivity = TP/(TP + FN)
    Precision = TP/(TP + FP)
    result = [Accuracy, Sensitivity, Precision]
    return data, result


def Baseline2(data_actual, data_na, location):
    # random number based on row
    actual = []
    pred = []
    data = data_na.copy()
    total = data.count(axis = 1)
    total = total.fillna(700)
    prob = data.sum(axis = 1, skipna = True)/total
    for i in range(len(location)):
        row = location[i][0]
        col = location[i][1]
        p = prob[row]
        predict = np.random.binomial(1, p, size = 1)
        data.loc[row, col] = predict
        actual.append(data_actual.loc[row, col])
        pred.append(predict)
    cm_test = confusion_matrix(actual, pred)
    TP = cm_test[1, 1]
    TN = cm_test[0, 0]
    FP = cm_test[0, 1]
    FN = cm_test[1, 0]
    Accuracy = (TP + TN)/(TP + FN + TN + FP)
    Sensitivity = TP/(TP + FN)
    Precision = TP/(TP + FP)
    result = [Accuracy, Sensitivity, Precision]
    return data, result
        

def mf(data_actual, data_na, location, components=5):
    data, result_temp = Baseline1(data_actual, data_na, location)
    X = np.array(data)
    model = NMF(n_components=components)
    W = model.fit_transform(X)
    H = model.components_
    predict = W@H
    actual = []
    pred = []
    for i in range(len(location)):
        row = location[i][0]
        col = location[i][1]
        r = data_actual.index.get_loc(row)
        c = data_actual.columns.get_loc(col)
        actual.append(data_actual.loc[row, col])
        pred.append(int(predict[r, c]>0.5))
        
    cm_test = confusion_matrix(actual, pred)
    
    TP = cm_test[1, 1]
    TN = cm_test[0, 0]
    FP = cm_test[0, 1]
    FN = cm_test[1, 0]
    Accuracy = (TP + TN)/(TP + FN + TN + FP)
    Sensitivity = TP/(TP + FN)
    Precision = TP/(TP + FP)
    return([Accuracy, Sensitivity, Precision])


def baseline_out(data0, proportion=0.02, n=20):
    general_Sensitivity = pd.DataFrame(np.ones((12, 2)),
                           columns = ['b1', 'b2'])
    general_Precision = pd.DataFrame(np.ones((12, 2)),
                           columns = ['b1', 'b2'])
    result_Sensitivity1 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Sensitivity2 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Precision1 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Precision2 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    for cuisine in range(12):
        for i in range(n):
            data = data0[data0['cuisine']==cuisine].iloc[:,:709].transpose() 
            output = data_generator_new(data, proportion)
            data_na = output[0]
            location = output[1]
            _, B1 = Baseline1(data, data_na, location)
            _, B2 = Baseline2(data, data_na, location)
            result_Sensitivity1.iloc[cuisine, i] = B1[1]
            result_Sensitivity2.iloc[cuisine, i] = B2[1]
            result_Precision1.iloc[cuisine, i] = B1[2]
            result_Precision2.iloc[cuisine, i] = B2[2]
            
    general_Sensitivity.iloc[:, 0] = result_Sensitivity1.mean(axis = 1)
    general_Sensitivity.iloc[:, 1] = result_Sensitivity2.mean(axis = 1)
    general_Precision.iloc[:, 0] = result_Precision1.mean(axis = 1)
    general_Precision.iloc[:, 1] = result_Precision2.mean(axis = 1)

    return general_Sensitivity, general_Precision


def pca_item_fit(data0, proportion=0.02, n=20):
    pc = list(range(20, 76, 5))
    general_Sensitivity = pd.DataFrame(np.ones((12, len(pc))),
                           columns = pc)
    general_Precision = pd.DataFrame(np.ones((12, len(pc))),
                           columns = pc)
    for p in range(len(pc)):
        result_Sensitivity = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        result_Precision = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        for cuisine in range(12):
            for i in range(n):
                data = data0[data0['cuisine']==cuisine].iloc[:,:709].transpose() 
                output = data_generator_new(data, proportion)
                data_na = output[0]
                location = output[1]
                sim = similarity_cosine_pca_item(data, data_na, location, pc[p])
                pred = UserItem_calculation1(data_na, location, sim, k=8)
                out = Metrics(data, pred, location)
                result_Sensitivity.iloc[cuisine, i] = out[1]
                result_Precision.iloc[cuisine, i] = out[2]
        general_Sensitivity.iloc[:, p] = result_Sensitivity.mean(axis = 1)
        general_Precision.iloc[:, p] = result_Precision.mean(axis = 1)
    
    return general_Sensitivity, general_Precision

def pca_user_fit(data0, proportion=0.02, n=20):
    pc = list(range(20, 51, 5))
    general_Sensitivity = pd.DataFrame(np.ones((12, len(pc))),
                           columns = pc)
    general_Precision = pd.DataFrame(np.ones((12, len(pc))),
                           columns = pc)
    for p in range(len(pc)):
        result_Sensitivity = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        result_Precision = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        for cuisine in range(12):
            for i in range(n):
                data = data0[data0['cuisine']==cuisine].iloc[:,:709].transpose() 
                output = data_generator_new(data, proportion)
                data_na = output[0]
                location = output[1]
                sim = similarity_cosine_pca_user(data, data_na, location, pc[p])
                pred = UserItem_calculation2(data_na, location, sim, k=8)
                out = Metrics(data, pred, location)
                result_Sensitivity.iloc[cuisine, i] = out[1]
                result_Precision.iloc[cuisine, i] = out[2]
        general_Sensitivity.iloc[:, p] = result_Sensitivity.mean(axis = 1)
        general_Precision.iloc[:, p] = result_Precision.mean(axis = 1)
    
    return general_Sensitivity, general_Precision


def mf_fit(data0, proportion=0.01, n=10):
    com = list(range(5, 61, 5))
    general_Sensitivity = pd.DataFrame(np.ones((12, len(com))),
                           columns = com)
    general_Precision = pd.DataFrame(np.ones((12, len(com))),
                           columns = com)
    for c in range(len(com)):
        result_Sensitivity = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        result_Precision = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        for cuisine in range(12):
            for i in range(n):
                data = data0[data0['cuisine']==cuisine].iloc[:,:709].transpose() 
                output = data_generator_new(data, proportion)
                data_na = output[0]
                location = output[1]
                out = mf(data, data_na, location, components=com[c])
                result_Sensitivity.iloc[cuisine, i] = out[1]
                result_Precision.iloc[cuisine, i] = out[2]
        general_Sensitivity.iloc[:, c] = result_Sensitivity.mean(axis = 1)
        general_Precision.iloc[:, c] = result_Precision.mean(axis = 1)
    
    return general_Sensitivity, general_Precision


def methods_out(data0, proportion=0.02, n=10):
    general_Sensitivity = pd.DataFrame(np.ones((12, 5)),
                           columns = ['item', 'user', 'pca_item', 'pca_user', 'mf'])
    general_Precision = pd.DataFrame(np.ones((12, 5)),
                           columns = ['item', 'user', 'pca_item', 'pca_user', 'mf'])
    result_Sensitivity1 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Sensitivity2 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Sensitivity3 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Sensitivity4 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Sensitivity5 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Precision1 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Precision2 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Precision3 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Precision4 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    result_Precision5 = pd.DataFrame(np.ones((12,n)), dtype = 'int')
    
    for cuisine in range(12):
        for i in range(n):
            data = data0[data0['cuisine']==cuisine].iloc[:,:709].transpose() 
            output = data_generator_new(data, proportion)
            data_na = output[0]
            location = output[1]
            sim1 = similarity_cosine_item(data, data_na, location)
            sim2 = similarity_cosine_user(data, data_na, location)
            sim3 = similarity_cosine_pca_item(data, data_na, location)
            sim4 = similarity_cosine_pca_user(data, data_na, location)
            
            pred1 = UserItem_calculation1(data_na, location, sim1, k=8)
            out1 = Metrics(data, pred1, location)
            pred2 = UserItem_calculation2(data_na, location, sim2, k=8)
            out2 = Metrics(data, pred2, location)
            pred3 = UserItem_calculation1(data_na, location, sim3, k=8)
            out3 = Metrics(data, pred3, location)
            pred4 = UserItem_calculation2(data_na, location, sim4, k=8)
            out4 = Metrics(data, pred4, location)
            
            result_Sensitivity1.iloc[cuisine, i] = out1[1]
            result_Sensitivity2.iloc[cuisine, i] = out2[1]
            result_Sensitivity3.iloc[cuisine, i] = out3[1]
            result_Sensitivity4.iloc[cuisine, i] = out4[1]
            result_Precision1.iloc[cuisine, i] = out1[2]
            result_Precision2.iloc[cuisine, i] = out2[2]
            result_Precision3.iloc[cuisine, i] = out3[2]
            result_Precision4.iloc[cuisine, i] = out4[2]
            
            out5 = mf(data, data_na, location)
            result_Sensitivity5.iloc[cuisine, i] = out5[1]
            result_Precision5.iloc[cuisine, i] = out5[2]
            
    general_Sensitivity.iloc[:, 0] = result_Sensitivity1.mean(axis = 1)
    general_Sensitivity.iloc[:, 1] = result_Sensitivity2.mean(axis = 1)
    general_Sensitivity.iloc[:, 2] = result_Sensitivity3.mean(axis = 1)
    general_Sensitivity.iloc[:, 3] = result_Sensitivity4.mean(axis = 1)
    general_Sensitivity.iloc[:, 4] = result_Sensitivity5.mean(axis = 1)
    general_Precision.iloc[:, 0] = result_Precision1.mean(axis = 1)
    general_Precision.iloc[:, 1] = result_Precision2.mean(axis = 1)
    general_Precision.iloc[:, 2] = result_Precision3.mean(axis = 1)
    general_Precision.iloc[:, 3] = result_Precision4.mean(axis = 1)
    general_Precision.iloc[:, 4] = result_Precision5.mean(axis = 1)

    return general_Sensitivity, general_Precision

def pca_item_NArate(data0, n=10):
    rate = [1, 2, 5, 10, 20]
    general_Sensitivity = pd.DataFrame(np.ones((12, len(rate))),
                           columns = rate)
    general_Precision = pd.DataFrame(np.ones((12, len(rate))),
                           columns = rate)
    for r in range(len(rate)):
        result_Sensitivity = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        result_Precision = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        for cuisine in range(12):
            for i in range(n):
                prop = rate[r]/100
                data = data0[data0['cuisine']==cuisine].iloc[:,:709].transpose() 
                output = data_generator_new(data, prop)
                data_na = output[0]
                location = output[1]
                sim = similarity_cosine_pca_item(data, data_na, location, 50)
                pred = UserItem_calculation1(data_na, location, sim, k=8)
                out = Metrics(data, pred, location)
                result_Sensitivity.iloc[cuisine, i] = out[1]
                result_Precision.iloc[cuisine, i] = out[2]
        general_Sensitivity.iloc[:, r] = result_Sensitivity.mean(axis = 1)
        general_Precision.iloc[:, r] = result_Precision.mean(axis = 1)
    
    return general_Sensitivity, general_Precision

def mf_NArate(data0, n=10):
    rate = [1, 2, 5, 10, 20]
    general_Sensitivity = pd.DataFrame(np.ones((12, len(rate))),
                           columns = rate)
    general_Precision = pd.DataFrame(np.ones((12, len(rate))),
                           columns = rate)
    for r in range(len(rate)):
        result_Sensitivity = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        result_Precision = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        for cuisine in range(12):
            for i in range(n):
                prop = rate[r]/100
                data = data0[data0['cuisine']==cuisine].iloc[:,:709].transpose() 
                output = data_generator_new(data, prop)
                data_na = output[0]
                location = output[1]
                out = mf(data, data_na, location, components=5)
                result_Sensitivity.iloc[cuisine, i] = out[1]
                result_Precision.iloc[cuisine, i] = out[2]
        general_Sensitivity.iloc[:, r] = result_Sensitivity.mean(axis = 1)
        general_Precision.iloc[:, r] = result_Precision.mean(axis = 1)
    
    return general_Sensitivity, general_Precision

def item_NArate(data0, n=10):
    rate = [1, 2, 5, 10, 20]
    general_Sensitivity = pd.DataFrame(np.ones((12, len(rate))),
                           columns = rate)
    general_Precision = pd.DataFrame(np.ones((12, len(rate))),
                           columns = rate)
    for r in range(len(rate)):
        result_Sensitivity = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        result_Precision = pd.DataFrame(np.ones((12, n)), dtype = 'int')
        for cuisine in range(12):
            for i in range(n):
                prop = rate[r]/100
                data = data0[data0['cuisine']==cuisine].iloc[:,:709].transpose() 
                output = data_generator_new(data, prop)
                data_na = output[0]
                location = output[1]
                sim = similarity_cosine_item(data, data_na, location)
                pred = UserItem_calculation1(data_na, location, sim, k=8)
                out = Metrics(data, pred, location)
                result_Sensitivity.iloc[cuisine, i] = out[1]
                result_Precision.iloc[cuisine, i] = out[2]
        general_Sensitivity.iloc[:, r] = result_Sensitivity.mean(axis = 1)
        general_Precision.iloc[:, r] = result_Precision.mean(axis = 1)
    
    return general_Sensitivity, general_Precision

def test_generator_1(data_actual, column, k):
    df = data_actual.copy()
    test = df.loc[:, column]
    idx1 = np.array(test[test == 1].index)
    idx2 = np.array(test[test == 0].index)
    choice = np.random.choice(idx1, k, replace=False)
    unknown = np.hstack((idx2, choice))
    df.loc[unknown, column] = np.nan
    return df, unknown

def test_generator_2(data_actual, column, k):
    df = data_actual.copy()
    #column = df.columns
    test = df.loc[:, column]
    idx1 = np.array(test[test == 1].index)
    idx2 = np.array(test[test == 0].index)
    choice1 = np.random.choice(idx1, k, replace=False)
    choice2 = np.random.choice(idx2, int(0.02*len(idx2)), replace=False)
    unknown = np.hstack((choice1, choice2))
    df.loc[unknown, column] = np.nan
    return df, unknown

def Baseline1_naive(data_actual, data_na, unknown, column):
    df_na = data_na.copy()
    data = df_na.loc[unknown, :]
    actual = data_actual.loc[unknown, column]
    pred = np.array(data.mode(axis = 1, dropna=True))
    if pred.shape[1] > 1:
        pred = pred[:, 0]
    df_na.loc[unknown, column] = pred
    cm_test = confusion_matrix(actual, pred)
    TP = cm_test[1, 1]
    TN = cm_test[0, 0]
    FP = cm_test[0, 1]
    FN = cm_test[1, 0]
    Accuracy = (TP + TN)/(TP + FN + TN + FP)
    Sensitivity = TP/(TP + FN)
    if TP+FP == 0:
        Precision = 0
    else:
        Precision = TP/(TP + FP)
    result = [Accuracy, Sensitivity, Precision]
    return df_na, result

def Baseline2_naive(data_actual, data_na, unknown, column):
    df_na = data_na.copy()
    data = df_na.loc[unknown, :]
    actual = data_actual.loc[unknown, column]
    total = data.count(axis = 1)
    prob = data.sum(axis = 1, skipna = True)/total
    pred = np.random.binomial(1, prob)
    df_na.loc[unknown, column] = pred
    cm_test = confusion_matrix(actual, pred)
    TP = cm_test[1, 1]
    TN = cm_test[0, 0]
    FP = cm_test[0, 1]
    FN = cm_test[1, 0]
    Accuracy = (TP + TN)/(TP + FN + TN + FP)
    Sensitivity = TP/(TP + FN)
    if TP+FP == 0:
        Precision = 0
    else:
        Precision = TP/(TP + FP)
    result = [Accuracy, Sensitivity, Precision]
    return df_na, result

def item_cosine_similarity_naive(data_actual, data_na, unknown, column, baseline, m):
    if (baseline == 1):
        data, result_temp = Baseline1_naive(data_actual, data_na, unknown, column)
    if (baseline == 2):
        data, result_temp = Baseline2_naive(data_actual, data_na, unknown, column)

    col = data_actual.columns
    actual = data_actual.loc[unknown, column]
    test = np.array(data.loc[:, column])
    train_col = col[col != column]
    train = data.loc[:, train_col].transpose()
    train_array = np.array(train)

    sim = cosine_similarity(test.reshape((1, -1)), train_array).reshape(-1)
    sim = pd.DataFrame(sim)
    choice_idx = sim.nlargest(m, 0).index.tolist()
    choice = col[choice_idx]
    pred = train.loc[choice, unknown].mean(axis=0).round(0)
    cm_test = confusion_matrix(actual, pred)

    TP = cm_test[1, 1]
    TN = cm_test[0, 0]
    FP = cm_test[0, 1]
    FN = cm_test[1, 0]
    Accuracy = (TP + TN) / (TP + FN + TN + FP)
    Sensitivity = TP / (TP + FN)
    if TP + FP == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    return ([Accuracy, Sensitivity, Precision])

def item_pca_naive(data_actual, data_na, unknown, column, baseline, m = 8):
    if (baseline == 1):
        data, result_temp = Baseline1_naive(data_actual, data_na, unknown, column)
    if (baseline == 2):
        data, result_temp = Baseline2_naive(data_actual, data_na, unknown, column)

    actual = data_actual.loc[unknown, column]
    data = data.transpose()
    pca = PCA(50)
    pca.fit(data)
    pc_scores = pd.DataFrame(pca.transform(data))
    col = data_actual.columns

    test_idx = col.get_loc(column)
    test = np.array(pc_scores.loc[test_idx, :])

    train = data.drop(column)
    train_pca = pc_scores.drop([test_idx])
    train_array = np.array(train_pca)

    sim = cosine_similarity(test.reshape((1, -1)), train_array).reshape(-1)
    sim = pd.DataFrame(sim)
    choice_idx = sim.nlargest(m, 0).index.tolist()
    choice = train.index[choice_idx]
    pred = train.loc[choice, unknown].mean(axis=0).round(0)
    cm_test = confusion_matrix(actual, pred)

    TP = cm_test[1, 1]
    TN = cm_test[0, 0]
    FP = cm_test[0, 1]
    FN = cm_test[1, 0]
    Accuracy = (TP + TN) / (TP + FN + TN + FP)
    Sensitivity = TP / (TP + FN)
    if TP + FP == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    return ([Accuracy, Sensitivity, Precision])

def mf_naive(data_actual, data_na, unknown, column, baseline, components):
    if (baseline == 1):
        data, result_temp = Baseline1_naive(data_actual, data_na, unknown, column)
    if (baseline == 2):
        data, result_temp = Baseline2_naive(data_actual, data_na, unknown, column)

    X = np.array(data)
    model = NMF(n_components=components, random_state=0, max_iter=400)
    W = model.fit_transform(X)
    H = model.components_
    predict = W @ H
    predict = pd.DataFrame(predict, columns = data_na.columns.tolist(), index = data_na.index.tolist())
    actual = data_actual.loc[unknown, column]
    pred = predict.loc[unknown, column].round(0)
    cm_test = confusion_matrix(actual, pred)

    TP = cm_test[1, 1]
    TN = cm_test[0, 0]
    FP = cm_test[0, 1]
    FN = cm_test[1, 0]
    Accuracy = (TP + TN) / (TP + FN + TN + FP)
    Sensitivity = TP / (TP + FN)
    if TP+FP == 0:
        Precision = 0
    else:
        Precision = TP/(TP + FP)
    return ([Accuracy, Sensitivity, Precision])

def Baseline1_naive_fit(recipes_all, prob = 0.1, na_choice = 1):
    result_b1 = pd.DataFrame(np.ones((12, 3)), columns = ["Accuracy", "Sensitivity", "Precision"])
    for c in range(12):
        sel = recipes_all.loc[recipes_all['cuisine'] == c].drop('cuisine', axis=1).transpose()
        r1, r2, r3 = [], [], []
        for i in range(0, sel.shape[1]):
            data_actual = sel.copy()
            col = data_actual.columns
            k = int(prob * data_actual.iloc[:, i].sum()) + 1

            column = col[i]
            if na_choice == 1:
                data_na, unknown = test_generator_1(data_actual, column, k)
            else:
                data_na, unknown = test_generator_2(data_actual, column, k)

            data_fill1, result_baseline1 = Baseline1_naive(data_actual, data_na, unknown, column)
            r1.append(result_baseline1[0])
            r2.append(result_baseline1[1])
            r3.append(result_baseline1[2])
        result_b1.loc[c, :] = np.array([np.mean(r1), np.mean(r2), np.mean(r3)])
    return result_b1

def Baseline2_naive_fit(recipes_all, prob = 0.1, na_choice = 1):
    result_b2 = pd.DataFrame(np.ones((12, 3)), columns = ["Accuracy", "Sensitivity", "Precision"])
    for c in range(12):
        sel = recipes_all.loc[recipes_all['cuisine'] == c].drop('cuisine', axis=1).transpose()
        r1, r2, r3 = [], [], []
        for i in range(0, sel.shape[1]):
            data_actual = sel.copy()
            col = data_actual.columns
            k = int(prob * data_actual.iloc[:, i].sum()) + 1

            column = col[i]
            if na_choice == 1:
                data_na, unknown = test_generator_1(data_actual, column, k)
            else:
                data_na, unknown = test_generator_2(data_actual, column, k)

            data_fill2, result_baseline2 = Baseline2_naive(data_actual, data_na, unknown, column)
            r1.append(result_baseline2[0])
            r2.append(result_baseline2[1])
            r3.append(result_baseline2[2])
        result_b2.loc[c, :] = np.array([np.mean(r1), np.mean(r2), np.mean(r3)])
    return result_b2

def MF_naive_fit(recipes_all, prob = 0.1, baseline = 1, components = 5, na_choice = 1):
    result_MF = pd.DataFrame(np.ones((12, 3)), columns = ["Accuracy", "Sensitivity", "Precision"])
    for c in range(12):
        sel = recipes_all.loc[recipes_all['cuisine'] == c].drop('cuisine', axis=1).transpose()
        r1, r2, r3 = [], [], []
        for i in range(0, sel.shape[1]):
            data_actual = sel.copy()
            col = data_actual.columns
            k = int(prob * data_actual.iloc[:, i].sum()) + 1

            column = col[i]
            if na_choice == 1:
                data_na, unknown = test_generator_1(data_actual, column, k)
            else:
                data_na, unknown = test_generator_2(data_actual, column, k)

            result_mf = mf_naive(data_actual, data_na, unknown, column, baseline, components)
            r1.append(result_mf[0])
            r2.append(result_mf[1])
            r3.append(result_mf[2])
        result_MF.loc[c, :] = np.array([np.mean(r1), np.mean(r2), np.mean(r3)])
    return result_MF

def item_pca_fit(recipes_all, prob = 0.1, baseline = 1, na_choice = 1):
    result_PCA = pd.DataFrame(np.ones((12, 3)), columns = ["Accuracy", "Sensitivity", "Precision"])
    for c in range(12):
        sel = recipes_all.loc[recipes_all['cuisine'] == c].drop('cuisine', axis=1).transpose()
        r1, r2, r3 = [], [], []
        for i in range(0, sel.shape[1]):
            data_actual = sel.copy()
            col = data_actual.columns
            k = int(prob * data_actual.iloc[:, i].sum()) + 1

            column = col[i]
            if na_choice == 1:
                data_na, unknown = test_generator_1(data_actual, column, k)
            else:
                data_na, unknown = test_generator_2(data_actual, column, k)

            result_pca = item_pca_naive(data_actual, data_na, unknown, column, baseline, m = 8)
            r1.append(result_pca[0])
            r2.append(result_pca[1])
            r3.append(result_pca[2])
        result_PCA.loc[c, :] = np.array([np.mean(r1), np.mean(r2), np.mean(r3)])
    return result_PCA