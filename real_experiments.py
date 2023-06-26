import pandas as pd
from SpecMix.benchmarks import calculate_score
import os
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")
from SpecMix.benchmarks import purity_score
from sklearn.preprocessing import LabelEncoder

def real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=',', drop = None, scaling = True):
    scores = {}
    avg_time_taken = {}
    

    # Load the CSV into a DataFrame
    df = pd.read_csv(path + filename, names=column_names, sep=sep)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    print(df)
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
    num_samples = df.shape[0]
    lambda_values = [0, 0.01, 10, 50, 100, 1000, num_samples/(num_clusters * len(categorical_cols))]

    le = LabelEncoder()
    #Turn categorical columns into object type
    for col in categorical_cols:
        df[col] = df[col].astype('object')
        df[col] = le.fit_transform(df[col])
        df[col] = df[col].astype('object')



    #Turn numerical columns into float or int type
    for col in numerical_cols:
        if df[col].dtype == 'object' or df[col].dtype == 'int64':
            df[col] = df[col].astype('float')

    for m in methods:
        print(m)
        if m == 'spectral':
            continue
        score, time_taken, _ = calculate_score(df, df['target'].tolist(), num_clusters, m, metrics=metrics, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
        scores[m] = score
        avg_time_taken[m] = time_taken
    if 'spectral' in methods:
        for ker in kernel:
            for l in lambda_values:
                print(f'spectral lambda={l} kernel={ker}')
                score, time_taken, _ = calculate_score(df, df['target'].tolist(), num_clusters, 'spectral',  metrics=metrics, lambdas=[l] * len(categorical_cols), numerical_cols=numerical_cols, categorical_cols=categorical_cols, kernel=ker, scaling = scaling)
                scores[f'spectral lambda={l} kernel={ker}'] = score
                print(score)
                avg_time_taken[f'spectral lambda={l} kernel={ker}'] = time_taken
    #Save scores and time taken
    #make directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/scores_scaling={scaling}.pkl', 'wb') as f:
        pkl.dump(scores, f)
    with open(f'{path}/avg_time_taken_scaling={scaling}.pkl', 'wb') as f:
        pkl.dump(avg_time_taken, f)
    
    #Save scores in csv
    scores_df = pd.DataFrame(scores)
    #first column is metrics
    scores_df.insert(0, 'metrics', metrics)
    scores_df.to_csv(f'{path}/scores_scaling={scaling}.csv', index=False)
    avg_time_taken_df = pd.DataFrame(avg_time_taken, index=[0])
    avg_time_taken_df.to_csv(f'{path}/avg_time_taken.csv', index=False)
    return scores, avg_time_taken

methods = ['spectral', 'famd', 'k-prototypes', 'lca', 'spectralCAT',  'onlyCat']
methods_scaling = ['spectral']
metrics = ['purity', 'calinski_harabasz', 'silhouette', 'adjusted_rand', 'homogeneity']
kernel = ['median_pairwise', 'preset']



# #Zoo
# path = 'Experiments/Real Datasets/Zoo/'
# filename = 'zoo.data'
# num_clusters = 7
# column_names = ['Animal Name', 'Hair', 'Feathers', 'Eggs', 'Milk', 'Airborne', 'Aquatic', 'Predator',
#                 'Toothed', 'Backbone', 'Breathes', 'Venomous', 'Fins', 'Legs', 'Tail', 'Domestic', 'Catsize', 'target']
# numerical_cols = ['Legs']
# categorical_cols = ['Hair', 'Feathers', 'Eggs', 'Milk', 'Airborne', 'Aquatic', 'Predator',
#                 'Toothed', 'Backbone', 'Breathes', 'Venomous', 'Fins', 'Tail', 'Domestic', 'Catsize']
# drop = ['Animal Name']
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop=drop, sep=',')
# print(scores)
# scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop=drop, sep=',', scaling=False)
# print(scores)

# #Heart Cleveland
# path = 'Experiments/Real Datasets/Heart Disease Cleveland/'
# filename = 'processed.cleveland.data'
# num_clusters = 5
# column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
#                 'oldpeak', 'slope', 'ca', 'thal', 'target']
# numerical_cols= ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
# categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=',')
# print(scores)
# scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=',', scaling=False)
# print(scores)

# #Dermatology
# path = 'Experiments/Real Datasets/Dermatology/'
# filename = 'dermatology.data'
# num_clusters = 6
# column_names = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 
#                 'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement',
#                 'scalp_involvement', 'family_history', 'melanin_incontinence', 'eosinophils_infiltrate',
#                 'PNL_infiltrate', 'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis',
#                 'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges',
#                 'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule', 'munro_microabcess',
#                 'focal_hypergranulosis', 'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer',
#                 'spongiosis', 'saw-tooth_appearance_of_retes', 'follicular_horn_plug', 'perifollicular_parakeratosis',
#                 'inflammatory_monoluclear_inflitrate', 'band-like_infiltrate', 'age', 'target']
# numerical_cols = ['age']
# categorical_cols = ['erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 
#                 'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement', 'knee_and_elbow_involvement',
#                 'scalp_involvement', 'family_history', 'melanin_incontinence', 'eosinophils_infiltrate',
#                 'PNL_infiltrate', 'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis',
#                 'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges', 'elongation_of_the_rete_ridges',
#                 'thinning_of_the_suprapapillary_epidermis', 'spongiform_pustule', 'munro_microabcess',
#                 'focal_hypergranulosis', 'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer',
#                 'spongiosis', 'saw-tooth_appearance_of_retes', 'follicular_horn_plug', 'perifollicular_parakeratosis',
#                 'inflammatory_monoluclear_inflitrate', 'band-like_infiltrate']
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=',')
# print(scores)
# scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=',', scaling=False)
# print(scores)

# Vowel
# path = 'Experiments/Real Datasets/Vowel/'
# filename = 'vowel-context.data'
# num_clusters = 11
# column_names = ['Train_Test', 'Speaker_Number', 'Sex', 'Feature0', 'Feature1', 'Feature2', 'Feature3',
#                 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'target']
# numerical_cols = ['Feature0', 'Feature1', 'Feature2', 'Feature3',
#                 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9']
# categorical_cols = ['Sex']
# drop = ['Train_Test', 'Speaker_Number']
# methods.remove('onlyCat')
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop=drop, sep=r"\s+")
# print(scores)
# scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=r"\s+", scaling=False)
# print(scores)


#Categorical Datasets
#Soybean
# path = 'Experiments/Real Datasets/Soybean/'
# filename = 'soybean-large.data'
# num_clusters = 19
# column_names = ['Date', 'Plant-stand', 'Precip', 'Temp', 'Hail', 'Crop-hist', 'Area-damaged', 'Severity', 
#                 'Seed-tmt', 'Germination', 'Plant-growth', 'Leaves', 'Leafspots-halo', 'Leafspots-marg', 
#                 'Leafspot-size', 'Leaf-shread', 'Leaf-malf', 'Leaf-mild', 'Stem', 'Lodging', 'Stem-cankers', 
#                 'Canker-lesion', 'Fruiting-bodies', 'External decay', 'Mycelium', 'Int-discolor', 
#                 'Sclerotia', 'Fruit-pods', 'Fruit spots', 'Seed', 'Mold-growth', 'Seed-discolor', 'Seed-size',
#                 'Shriveling', 'Roots', 'target']
# numerical_cols = []
# categorical_cols = ['Date', 'Plant-stand', 'Precip', 'Temp', 'Hail', 'Crop-hist', 'Area-damaged', 'Severity', 
#                 'Seed-tmt', 'Germination', 'Plant-growth', 'Leaves', 'Leafspots-halo', 'Leafspots-marg', 
#                 'Leafspot-size', 'Leaf-shread', 'Leaf-malf', 'Leaf-mild', 'Stem', 'Lodging', 'Stem-cankers', 
#                 'Canker-lesion', 'Fruiting-bodies', 'External decay', 'Mycelium', 'Int-discolor', 
#                 'Sclerotia', 'Fruit-pods', 'Fruit spots', 'Seed', 'Mold-growth', 'Seed-discolor', 'Seed-size',
#                 'Shriveling', 'Roots']
# methods_categorical = ['lca', 'onlyCat', 'k-modes', 'spectralCAT']
# drop = []
# scores, avg_time_taken = real_experiments(methods_categorical, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop=drop, sep=',')
# print(scores)

# # Mushroom
# path = 'Experiments/Real Datasets/Mushroom/'
# filename = 'agaricus-lepiota.data'
# num_clusters = 2
# column_names = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
#                 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
#                 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
#                 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
# numerical_cols = []
# categorical_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
#                 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
#                 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
#                 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
# scores, avg_time_taken = real_experiments(methods_categorical, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop=drop, sep=',')
# print(scores)

# #Lung Cancer
# # path = 'Experiments/Real Datasets/Lung Cancer/'
# # filename = 'lung-cancer.data'
# # num_clusters = 3
# # column_names = ["target"]
# # column_names.extend(["Attribute {}".format(i) for i in range(1, 57)])
# # print(column_names)
# # numerical_cols = []
# # categorical_cols = ["Attribute {}".format(i) for i in range(1, 57)]
# scores, avg_time_taken = real_experiments(methods_categorical, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop=drop, sep=',') 
# print(scores)

# # #Car Evaluation
# # path = 'Experiments/Real Datasets/Car Evaluation/'
# # filename = 'car.data'
# # num_clusters = 4
# # column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target']
# # numerical_cols = []
# # categorical_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
# scores, avg_time_taken = real_experiments(methods_categorical, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, drop=drop, sep=',')
# print(scores)


#Mixed Datasets
drop = []


# # Obesity
# path = 'Experiments/Real Datasets/Obesity/'
# filename = 'ObesityDataSet_raw_and_data_sinthetic.csv'
# num_clusters = 7
# # Column names
# column_names = ["Gender", "Age", "Height", "Weight", "Family history with overweight",
#                 "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF", 
#                 "TUE", "CALC", "MTRANS", "target"]

# # Numerical Columns
# numerical_columns = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# # Categorical Columns
# categorical_columns = ["Gender", "Family history with overweight", "FAVC", "CAEC",
#                        "SMOKE", "SCC", "CALC", "MTRANS"]
# print("Obesity")
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, drop=drop, sep=",")
# print(scores)
# scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, sep=",", scaling=False)
# print(scores)

# #Hepatitis
# path = 'Experiments/Real Datasets/Hepatitis/'
# filename = 'hepatitis.data'
# num_clusters = 2
# # Column names
# column_names = ["target", "Age", "Sex", "Steroid", "Antivirals", "Fatigue", 
#                 "Malaise", "Anorexia", "Liver big", "Liver firm", "Spleen palpable", 
#                 "Spiders", "Ascites", "Varices", "Bilirubin", "Alk phosphate", "SGOT", 
#                 "Albumin", "Protime", "Histology"]

# # Numerical Columns
# numerical_columns = ["Age", "Bilirubin", "Alk phosphate", "SGOT", "Albumin", "Protime"]

# # Categorical Columns
# categorical_columns = ["Sex", "Steroid", "Antivirals", "Fatigue", "Malaise", 
#                        "Anorexia", "Liver big", "Liver firm", "Spleen palpable", 
#                        "Spiders", "Ascites", "Varices", "Histology"]
# print("Hepatitis")
# # scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, drop=drop, sep=",")
# # print(scores)
# scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, sep=",", scaling=True)
# print(scores)

# #Cylinder Bands
# path = 'Experiments/Real Datasets/Cylinder Bands/'
# filename = 'bands.data'
# num_clusters = 2
# # Column names
# column_names = ["timestamp", "cylinder number", "customer", "job number", "grain screened",
#                 "ink color", "proof on ctd ink", "blade mfg", "cylinder division", 
#                 "paper type", "ink type", "direct steam", "solvent type", 
#                 "type on cylinder", "press type", "press", "unit number", 
#                 "cylinder size", "paper mill location", "plating tank", "proof cut",
#                 "viscosity", "caliper", "ink temperature", "humifity", "roughness", 
#                 "blade pressure", "varnish pct", "press speed", "ink pct", "solvent pct",
#                 "ESA Voltage", "ESA Amperage", "wax", "hardener", "roller durometer", 
#                 "current density", "anode space ratio", "chrome content", "target"]

# # Numerical Columns
# numerical_columns = ["timestamp", "job number", "press", "unit number", "proof cut",
#                      "viscosity", "caliper", "ink temperature", "humifity", "roughness",
#                      "blade pressure", "varnish pct", "press speed", "ink pct", 
#                      "solvent pct", "ESA Voltage", "ESA Amperage", "wax", "hardener", 
#                      "roller durometer", "current density", "anode space ratio", 
#                      "chrome content"]

# # Categorical Columns
# categorical_columns = ["cylinder number", "customer", "grain screened", "ink color", 
#                        "proof on ctd ink", "blade mfg", "cylinder division", "paper type", 
#                        "ink type", "direct steam", "solvent type", "type on cylinder", 
#                        "press type", "cylinder size", "paper mill location", "plating tank"]
# print("Cylinder Bands")
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, drop=drop, sep=",")
# print(scores)
# scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, sep=",", scaling=False)
# print(scores)

# #Post Operative Patient
# path = 'Experiments/Real Datasets/Post Operative Patient/'
# filename = 'post-operative.data'
# num_clusters = 3
# # Column names
# column_names = ["L-CORE", "L-SURF", "L-O2", "L-BP", "SURF-STBL", 
#                 "CORE-STBL", "BP-STBL", "COMFORT", "target"]

# # Numerical Columns
# numerical_columns = ["COMFORT"]

# # Categorical Columns
# categorical_columns = ["L-CORE", "L-SURF", "L-O2", "L-BP", 
#                        "SURF-STBL", "CORE-STBL", "BP-STBL"]
# print("Post Operative Patient")
# # scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, drop=drop, sep=",")
# # print(scores)

# scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, sep=",", scaling=False)
# print(scores)



# #Credit Approval
# path = 'Experiments/Real Datasets/Credit Approval/'
# filename = 'crx.data'
# num_clusters = 2
# # Column names
# column_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", 
#                 "A9", "A10", "A11", "A12", "A13", "A14", "A15", "target"]

# # Numerical Columns
# numerical_columns = ["A2", "A3", "A8", "A11", "A14", "A15"]

# # Categorical Columns
# categorical_columns = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
# print("Credit Approval")
# scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, drop=drop, sep=",")
# print(scores)
# scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, sep=",", scaling=False)
# print(scores)

#Student Dropout
path = 'Experiments/Real Datasets/Student Dropout/'
filename = 'data.csv'
num_clusters = 3
# Column names
column_names = ['Marital_Status', 'Application_Mode', 'Application_Order', "Course", "Daytime/evening_attendance", "Prev_qualification",
                'Nationality', 'Mother_Qualification', 'Father_Qualification',           
                'Mother_Occupation', 'Father_Occupation', 'Admission_Grade', 'Displaced', 'Education_Special_Needs', 'Debtor',
                'Tuition_Fees_Up_To_Date', 'Gender', 'Scholarship_Holder', 'Age_At_Enrollment', 'International', 'C1_credited',
                'C1_enrolled', 'C1_evaluations', 'C1_approved', 'C1_grade', 'C1_without_evaluations', 'C2_credited', 'C2_enrolled',
                'C2_evaluations', 'C2_approved', 'C2_grade', 'C2_without_evaluations', 'Unemployment_Rate', 'Inflation_Rate', 'GDP',
                  'target']

# Numerical Columns

numerical_columns = ['Admission_Grade', 'Age_At_Enrollment', 'C1_credited', 'C1_enrolled', 'C1_evaluations', 'C1_approved', 'C1_grade', 'C1_without_evaluations', 'C2_credited', 'C2_enrolled', 'C2_evaluations', 'C2_approved', 'C2_grade', 'C2_without_evaluations', 'Unemployment_Rate', 'Inflation_Rate', 'GDP']

# Categorical Columns
categorical_columns = list(set(column_names)-set(numerical_columns)-set(['target']))
print("Student Dropout")
scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, drop=drop, sep=";")
print(scores)
scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, sep=";", scaling=False)
print(scores)

#Adult
path = 'Experiments/Real Datasets/Adult/'
filename = 'adult.data'
num_clusters = 2
column_names = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital-Status', 'Occupation',
                'Relationship', 'Race', 'Sex', 'Capital-Gain', 'Capital-Loss', 'Hours-per-Week', 'Native-Country', 'target']
numerical_cols = ['Age', 'fnlwgt', 'Education-Num', 'Capital-Gain', 'Capital-Loss', 'Hours-per-Week']
categorical_cols = ['Workclass', 'Education', 'Marital-Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Native-Country']
scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_cols, numerical_cols, path, filename, sep=',')
print(scores)
scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, sep=",", scaling=False)
print(scores)

# Bank Marketing
path = 'Experiments/Real Datasets/Bank Marketing/bank/'
filename = 'bank-full.csv'
num_clusters = 2
# Column names
column_names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan",
                "contact", "day", "month", "duration", "campaign", "pdays",
                "previous", "poutcome", "target"]

# Numerical Columns
numerical_columns = ["age", "duration", "campaign", "pdays", "previous", 
                     "balance"]

# Categorical Columns
categorical_columns = ["job", "marital", "education", "default", "housing", 
                       "loan", "contact", "month", "day", "poutcome"]
print("Bank Marketing")
scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, drop=drop, sep=";")
print(scores)
scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, sep=";", scaling=False)
print(scores)

# #Multivariate Gait
path = 'Experiments/Real Datasets/Multivariate Gait/'
filename = 'gait.csv'
num_clusters = 10
# Column names
column_names = ['target', 'condition', 'replication', 'leg', 'joint', 'time', 'angle']

# Numerical Columns
numerical_columns = ['replication', 'time', 'angle']

# Categorical Columns
categorical_columns = ['condition', 'leg', 'joint']
print("Multivariate Gait")
scores, avg_time_taken = real_experiments(methods, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, drop=drop, sep=",")
print(scores)
scores, avg_time_taken = real_experiments(methods_scaling, metrics, num_clusters, kernel, column_names, categorical_columns, numerical_columns, path, filename, sep=",", scaling=False)
print(scores)