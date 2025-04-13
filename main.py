import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC

hrdf = pd.read_csv('.venv/train.csv')
print(hrdf.head())
print(hrdf.info())

hrdf['education'] = hrdf['education'].fillna("Bachelor's")
hrdf['previous_year_rating'] = hrdf['previous_year_rating'].fillna(3.0)

print(hrdf.columns)

numcols = hrdf[['no_of_trainings', 'age','length_of_service', 'avg_training_score']]
objcols = hrdf[['department', 'region', 'education', 'gender',
       'recruitment_channel', 'previous_year_rating', 'KPIs_met >80%', 'awards_won?', 'is_promoted']]

objcols_dummy = pd.get_dummies(objcols, columns=['department', 'region', 'education', 'gender',
       'recruitment_channel', 'previous_year_rating', 'KPIs_met >80%', 'awards_won?'])

hrdf_final = pd.concat([numcols, objcols_dummy], axis=1)
y = hrdf_final["is_promoted"]
X = hrdf_final.drop(['is_promoted'], axis=1)
print(objcols_dummy.columns)

smote = SMOTENC(categorical_features=['department_Analytics', 'department_Finance',
       'department_HR', 'department_Legal', 'department_Operations',
       'department_Procurement', 'department_R&D',
       'department_Sales & Marketing', 'department_Technology',
       'region_region_1', 'region_region_10', 'region_region_11',
       'region_region_12', 'region_region_13', 'region_region_14',
       'region_region_15', 'region_region_16', 'region_region_17',
       'region_region_18', 'region_region_19', 'region_region_2',
       'region_region_20', 'region_region_21', 'region_region_22',
       'region_region_23', 'region_region_24', 'region_region_25',
       'region_region_26', 'region_region_27', 'region_region_28',
       'region_region_29', 'region_region_3', 'region_region_30',
       'region_region_31', 'region_region_32', 'region_region_33',
       'region_region_34', 'region_region_4', 'region_region_5',
       'region_region_6', 'region_region_7', 'region_region_8',
       'region_region_9', "education_Bachelor's", 'education_Below Secondary',
       "education_Master's & above", 'gender_f', 'gender_m',
       'recruitment_channel_other', 'recruitment_channel_referred',
       'recruitment_channel_sourcing', 'previous_year_rating_1.0',
       'previous_year_rating_2.0', 'previous_year_rating_3.0',
       'previous_year_rating_4.0', 'previous_year_rating_5.0',
       'KPIs_met >80%_0', 'KPIs_met >80%_1', 'awards_won?_0', 'awards_won?_1'])

X_smote,y_smote = smote.fit_resample(X, y)
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(max_iter=2000).fit(X_smote, y_smote)
print(reg.score(X_smote, y_smote))

reg_predict = reg.predict(X_smote)
print(pd.crosstab(y_smote, reg_predict, rownames=['Actual'], colnames=['Predicted']))

import pickle
with open("reg_model.pkl", "wb") as f:
    pickle.dump(reg, f)

with open("col.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)