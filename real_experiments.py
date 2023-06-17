import pandas as pd
from SpecMix.benchmarks import calculate_score
import os
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")
from SpecMix.benchmarks import purity_score


def real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=',', drop = None):
    scores = {}
    avg_time_taken = {}
    lambda_values = [0, 1, 100, 1000]
    # if 'spectral' in methods:
    #     lambda_values = [0, 10, 100, 1000]
    #     lambda_keys = []
    #     if kernel:
    #         for ker in kernel:
    #             lambda_keys.extend([f'spectral lambda={l} kernel={ker}' for l in lambda_values])               
    #     else:
    #         lambda_keys = [f'spectral lambda={l}' for l in lambda_values]
    #     # for k in lambda_keys:
    #     #     for metric in metrics:
    #     #         scores[metric][k] = 0
    #     #     avg_time_taken[k] = 0
    # # del scores['spectral']
    # # del avg_time_taken['spectral']

    # Load the CSV into a DataFrame
    df = pd.read_csv(path + filename, names=column_names, sep=sep)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop the specified columns
    if drop:
        df = df.drop(drop, axis=1)
    # Replace '?' with NaN
    df.replace('?', pd.NA, inplace=True)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    df['target'] = df['target'].astype('category').cat.codes

    print(df)
    #Turn categorical columns into object type
    for col in categorical_cols:
        df[col] = df[col].astype('object')

    #Turn numerical columns into float or int type
    for col in numerical_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('float')
    for m in methods:
        if m == 'spectral':
            continue
        score, time_taken, _ = calculate_score(df, df['target'].tolist(), num_clusters, m, metrics=metrics, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
        scores[m] = score
        avg_time_taken[m] = time_taken
    if 'spectral' in methods:
        for ker in kernel:
            curr_kernel = 0
            for l in lambda_values:
                if kernel:
                    if not curr_kernel:
                        score, time_taken, curr_kernel = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=[l] * len(categorical_cols), numerical_cols=numerical_cols, categorical_cols=categorical_cols, kernel=ker)
                    else:
                        score, time_taken, curr_kernel = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=[l] * len(categorical_cols), numerical_cols=numerical_cols, categorical_cols=categorical_cols, kernel=None, curr_kernel=curr_kernel)
                    scores[f'spectral lambda={l} kernel={ker}'] = score
                    avg_time_taken[f'spectral lambda={l} kernel={ker}'] = time_taken
                else:
                    lambdas = [l] * len(categorical_cols)
                    score, time_taken, _ = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=lambdas, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
                    scores[f'spectral lambda={l}'] = score
                    avg_time_taken[f'spectral lambda={l}'] = time_taken
            curr_kernel = 0
        
    #Save scores and time taken
    #make directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/scores.pkl', 'wb') as f:
        pkl.dump(scores, f)
    with open(f'{path}/avg_time_taken.pkl', 'wb') as f:
        pkl.dump(avg_time_taken, f)
    
    #Save scores in csv
    scores_df = pd.DataFrame(scores)
    #first column is metrics
    scores_df.insert(0, 'metrics', metrics)
    scores_df.to_csv(f'{path}/scores.csv', index=False)
    avg_time_taken_df = pd.DataFrame(avg_time_taken, index=[0])
    avg_time_taken_df.to_csv(f'{path}/avg_time_taken.csv', index=False)
    return scores, avg_time_taken

methods = ['spectral', 'k-prototypes', 'lca', 'spectralCAT' ]
metrics = ['purity', 'calinski_harabasz', 'silhouette', 'adjusted_rand', 'homogeneity']
num_clusters = 3
kernel = ['median_pairwise', 'cv_sigma', 'preset']


#Adult
# path = 'Experiments/Real Datasets/Adult/'
# filename = 'adult.data'
# num_clusters = 2
# column_names = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital-Status', 'Occupation',
#                 'Relationship', 'Race', 'Sex', 'Capital-Gain', 'Capital-Loss', 'Hours-per-Week', 'Native-Country', 'target']
# numerical_cols = ['Age', 'fnlwgt', 'Education-Num', 'Capital-Gain', 'Capital-Loss', 'Hours-per-Week']
# categorical_cols = ['Workclass', 'Education', 'Marital-Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Native-Country']
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=',')
# print(scores)
#Zoo
path = 'Experiments/Real Datasets/Zoo/'
filename = 'zoo.data'
num_clusters = 7
column_names = ['Animal Name', 'Hair', 'Feathers', 'Eggs', 'Milk', 'Airborne', 'Aquatic', 'Predator',
                'Toothed', 'Backbone', 'Breathes', 'Venomous', 'Fins', 'Legs', 'Tail', 'Domestic', 'Catsize', 'target']
numerical_cols = ['Legs']
categorical_cols = ['Hair', 'Feathers', 'Eggs', 'Milk', 'Airborne', 'Aquatic', 'Predator',
                'Toothed', 'Backbone', 'Breathes', 'Venomous', 'Fins', 'Tail', 'Domestic', 'Catsize']
drop = ['Animal Name']
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop=drop, sep=',')
# print(scores)
methods.append('famd')

#Heart Cleveland
path = 'Experiments/Real Datasets/Heart Disease Cleveland/'
filename = 'processed.cleveland.data'
num_clusters = 5
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                'oldpeak', 'slope', 'ca', 'thal', 'target']
numerical_cols= ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=',')
# print(scores)

#Dermatology
path = 'Experiments/Real Datasets/Dermatology/'
filename = 'dermatology.data'
num_clusters = 6
column_names = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 
                'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement',
                'scalp_involvement', 'family_history', 'melanin_incontinence', 'eosinophils_infiltrate',
                'PNL_infiltrate', 'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis',
                'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges',
                'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule', 'munro_microabcess',
                'focal_hypergranulosis', 'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer',
                'spongiosis', 'saw-tooth_appearance_of_retes', 'follicular_horn_plug', 'perifollicular_parakeratosis',
                'inflammatory_monoluclear_inflitrate', 'band-like_infiltrate', 'age', 'target']
numerical_cols = ['age']
categorical_cols = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 
                'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement',
                'scalp_involvement', 'family_history', 'melanin_incontinence', 'eosinophils_infiltrate',
                'PNL_infiltrate', 'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis',
                'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges',
                'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule', 'munro_microabcess',
                'focal_hypergranulosis', 'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer',
                'spongiosis', 'saw-tooth_appearance_of_retes', 'follicular_horn_plug', 'perifollicular_parakeratosis',
                'inflammatory_monoluclear_inflitrate', 'band-like_infiltrate']
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=',')
# print(scores)
#Vowel
path = 'Experiments/Real Datasets/Vowel/'
filename = 'vowel-context.data'
num_clusters = 11
column_names = ['Train_Test', 'Speaker_Number', 'Sex', 'Feature0', 'Feature1', 'Feature2', 'Feature3',
                'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'target']
numerical_cols = ['Feature0', 'Feature1', 'Feature2', 'Feature3',
                'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9']
categorical_cols = ['Sex']
drop = ['Train_Test', 'Speaker_Number']
scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop=drop, sep=r"\s+")
print(scores)






path = 'Experiments/Real Datasets/Iris/'
# filename = 'iris.data'
# column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
# categorical_cols = []
# numerical_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# #scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename)
# #print(scores)

# #Wine

# path = 'Experiments/Real Datasets/Wine/'
# filename = 'wine.data'
# column_names  = ["target", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "date", "proline"]
# numerical_cols = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "date", "proline"]
# #scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename)
# # print(scores)


# #Glass 
# path = 'Experiments/Real Datasets/Glass/'
# filename = 'glass.data'
# num_clusters = 6
# column_names = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'target']
# numerical_cols = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
# drop = ['id']
# #scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop=drop)
# #print(scores)
# #

#Segmentation
# path = 'Experiments/Real Datasets/Segmentation/'
# filename = 'segmentation.data'
# num_clusters = 7
# column_names = ['target', 'REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2',
#                 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN', 'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN',
#                 'EXRED-MEAN', 'EXBLUE-MEAN', 'EXGREEN-MEAN', 'VALUE-MEAN', 'SATURATION-MEAN', 'HUE-MEAN']
# numerical_cols = ['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2',
#                 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN', 'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN',
#                 'EXRED-MEAN', 'EXBLUE-MEAN', 'EXGREEN-MEAN', 'VALUE-MEAN', 'SATURATION-MEAN', 'HUE-MEAN']
# drop = ['REGION-PIXEL-COUNT']
# #scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop = drop, sep=',')
# #print(scores)

# #Ionosphere
# path = 'Experiments/Real Datasets/Ionosphere/'
# filename = 'ionosphere.data'
# num_clusters = 2
# column_names = ['Attribute'+str(i) for i in range(1, 35)] + ['target']
# numerical_cols = ['Attribute'+str(i) for i in range(1, 35)]
# drop = ['Attribute2']
# numerical_cols.remove('Attribute2')
# # scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop = drop, sep=',')
# # print(scores)
