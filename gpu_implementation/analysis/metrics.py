import numpy as np

def f_is_feasible(inputData):
    """
    :param inputData: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_feasible = np.ones(inputData.shape[0], dtype = bool)
    for i, c in enumerate(inputData):
        is_feasible[i] = np.all(inputData[i,:]>0)#prima era: (c>0)
    return is_feasible
def f_is_efficient(inputData):
        """
        :param inputData: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(inputData.shape[0], dtype = bool)
        for i, c in enumerate(inputData):
            d = inputData[i,:]
            is_efficient[i] = not np.any(np.all( [ (np.all(inputData>=d, axis=1)) , (np.any(inputData>d, axis=1)) ] , axis = 0) )
        return is_efficient
def f_feasible_points(inputData):
    is_feasible = f_is_feasible(inputData)
    x = inputData[0,:]
    y = inputData[1,:]
    i = 0
    x_feasible = []
    y_feasible = []
    while i<len(inputData):
        if is_feasible[i]:
            x_feasible.append(x[i])
            y_feasible.append(y[i])
        i += 1
    return x_feasible, y_feasible
def f_ordered_PF_points(inputData):
    x_eff = inputData[0,:]
    y_eff = inputData[1,:]

    x_aus = x_eff.copy()
    y_aus = y_eff.copy()
    i = 0
    while i<len(x_eff)-1:
        j = i+1
        while j < len(x_eff):
            if x_eff[j]<x_eff[i]:
                x_eff[i] = x_eff[j]
                x_eff[j] = x_aus[i]
                x_aus = x_eff.copy()
                y_eff[i] = y_eff[j]
                y_eff[j] = y_aus[i]
                y_aus = y_eff.copy()
            j+=1
        i += 1
    return x_eff, y_eff


def f_true_PF_points(inputData):
    is_efficient = f_is_efficient(inputData)
    is_feasible = f_is_feasible(inputData)
    x = inputData[:,0] 
    y = inputData[:,1]
    i = 0
    x_eff = []
    y_eff = []
    while i<len(inputData):
        if is_efficient[i] and is_feasible[i]:
            x_eff.append(x[i])
            y_eff.append(y[i])
        i += 1
    x_eff, y_eff = f_ordered_PF_points(np.array([x_eff,y_eff]))
    return x_eff,y_eff

def f_computeHypervolume(inputData):
    HV = inputData[0][0]*inputData[1][0]
    i = 1
    while i<len(inputData[0])-1:
        HV += (inputData[0][i]-inputData[0][i-1])*inputData[1][i]
        i+=1
    return HV
