import warnings
import os
from cf_module import*

#The supported functions are included in cf_module

warnings.filterwarnings('ignore')
data_path = os.path.join(os.getcwd(),"recipes.csv")
recipes_all = pd.read_csv(data_path, delimiter = ',')
data = recipes_all.copy()


#4.2
#determine how many PC should we keep for pca-item and pca-user
pca_item_Sensitivity, pca_item_Precision = pca_item_fit(data, proportion = 0.02, n = 20)
pca_user_Sensitivity, pca_user_Precision = pca_user_fit(data, proportion = 0.02, n = 20)
mf_Sensitivity, mf_Precision = mf_fit(data0, proportion=0.02, n=20)


#4.2 Model comparison
#1. result for two baselines
baseline_Sensitivity, baseline_Precision = baseline_out(data, proportion = 0.02, n = 20)
#2. result for all models
all_Sensitivity, all_Precision = methods_out(data, proportion = 0.02, n=20)


#4.3 Effect of missing proportion
pca_Sensitivity, pca_Precision = pca_item_NArate(data, n = 20)
mf_Sensitivity, mf_Precision = mf_NArate(data, n =20)


#4.4 CANs and CAN’Ts
#For the na selection method where only one recipe is largely unknown
#baseline 1: mode of every ingredient
result_b1 = Baseline1_naive_fit(data, prob = 0.2, na_choice = 1)
#baseline 2: binomial random variable
result_b2 = Baseline2_naive_fit(data, prob = 0.2, na_choice = 1)
#item-based PCA
result_pca = item_pca_fit(data0, prob = 0.2, baseline = 1, na_choice = 1)
#Matrix Factorization
result_mf = MF_naive_fit(data0, prob = 0.2, baseline = 1, components = 5, na_choice = 1)