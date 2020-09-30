import scipy.io as spio
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
def pass_arg(Xx, nsim, tr_size):

    print("tr_Size:",tr_size)
    tr_size = int(tr_size)
    use_YPhy = 0

    #List of lakes to choose from
    lake = ['mendota' , 'mille_lacs']
    lake_num = 0  # 0 : mendota , 1 : mille_lacs
    lake_name = lake[lake_num]

    # Load features (Xc) and target values (Y)
    data_dir = '../../../../data/'
    filename = lake_name + '.mat'
    mat = spio.loadmat(data_dir + filename, squeeze_me=True,
    variable_names=['Y','Xc_doy','Modeled_temp'])
    Xc = mat['Xc_doy']
    Y = mat['Y']
    Xc = Xc[:,:-1]
    # train and test data
    trainX, testX, trainY, testY = train_test_split(Xc, Y, train_size=tr_size/Xc.shape[0], 
                                                    test_size=tr_size/Xc.shape[0], random_state=42, shuffle=True)

    ## train and test data
    #trainX, trainY = Xc[:tr_size,:-1], Y[:tr_size]
    #testX, testY = Xc[-50:,:-1], Y[-50:]
    
    kernel = C(5.0, (0.5, 1e1)) * RBF(length_scale = [1] * trainX.shape[1], length_scale_bounds=(1e-1, 1e7))
    gp = GaussianProcessRegressor(kernel=kernel, alpha =1.5, n_restarts_optimizer=10)
    gp.fit(trainX, trainY)
    #y_pred1, sigma1 = gp.predict(testX, return_std=True)
    
    # scale the uniform numbers to original space
    # max and min value in each column 
    max_in_column_Xc = np.max(trainX,axis=0)
    min_in_column_Xc = np.min(trainX,axis=0)
        
    # Xc_scaled = (Xc-min_in_column_Xc)/(max_in_column_Xc-min_in_column_Xc)
    Xc_org = Xx*(max_in_column_Xc-min_in_column_Xc) + min_in_column_Xc
        
        
    samples = gp.sample_y(Xc_org, n_samples=int(nsim)).T
    return samples