from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
from sklearn.ensemble import RandomForestClassifier

import os
import openai
import torch
from caafe import data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn.scripts import tabular_metrics
from functools import partial
import openml
import pandas as pd
import numpy as np

# benchmark_ids = []
# suite = openml.study.get_suite(271) # Classification
# # suite = openml.study.get_suite(269) # Regression
# tasks = suite.tasks
# for task_id in tasks:
#     task = openml.tasks.get_task(task_id)
#     datasetID = task.dataset_id
#     benchmark_ids.append(datasetID)

openai.api_key = ""

metric_used = tabular_metrics.auc_metric
dataset = openml.datasets.get_dataset(41143)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

df_train = pd.DataFrame(
        data=np.concatenate([X_train, np.expand_dims(y_train, -1)], -1), columns=attribute_names + [dataset.default_target_attribute]
    )
df_test = pd.DataFrame(
        data=np.concatenate([X_test, np.expand_dims(y_test, -1)], -1), columns=attribute_names + [dataset.default_target_attribute]
    )
# cc_test_datasets_multiclass = data.load_all_data(benchmark_ids[65:67]) # We consider two as an example
# cc_test_datasets_multiclass = data.load_all_data(['41143']) # We consider two as an example

# ds = cc_test_datasets_multiclass[0] # let's get the first dataset from the largest ones
# ds, df_train, df_test, _, _ = data.get_data_split(ds, seed=0)
# target_column_name = ds[4][-1]
# dataset_description = ds[-1]
# ds[0]

from caafe.preprocessing import make_datasets_numeric
#, df_test = make_datasets_numeric(df_train, df_test, target_column_name)
# Replace this with a function to reduce the dimension of the dataset in case features are up > 100
# Making a simple pipeline for categorical features
# train_x, train_y = data.get_X_y(df_train, target_column_name)
# test_x, test_y = data.get_X_y(df_test, target_column_name)


### Setup Base Classifier

clf_no_feat_eng = RandomForestClassifier()
# clf_no_feat_eng = TabPFNClassifier(device=('cuda' if torch.cuda.is_available() else 'cpu'), N_ensemble_configurations=4)
# clf_no_feat_eng.fit = partial(clf_no_feat_eng.fit, overwrite_warning=True)

clf_no_feat_eng.fit(X_train, y_train)
pred = clf_no_feat_eng.predict(X_test)
acc = accuracy_score(pred, y_test)
print(f'Accuracy before CAAFE {acc}')

### Setup and Run CAAFE - This will be billed to your OpenAI Account!

caafe_clf = CAAFEClassifier(base_classifier=clf_no_feat_eng,
                            llm_model="gpt-3.5-turbo",
                            iterations=2)

# The iterations happen here:
caafe_clf.fit_pandas(df_train,
                     target_column_name=dataset.default_target_attribute,
                     dataset_description=dataset.description)

# This process is done only once
pred = caafe_clf.predict(df_test)
acc = accuracy_score(pred, y_test)
print(f'Accuracy after CAAFE {acc}')