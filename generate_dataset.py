from caafe import CAAFEClassifier # Automated Feature Engineering for tabular datasets
from tabpfn import TabPFNClassifier # Fast Automated Machine Learning method for small tabular datasets
from sklearn.ensemble import RandomForestClassifier

import os
import openai
import openml
import torch
from caafe import data, create_prompt
from sklearn.metrics import accuracy_score
from tabpfn.scripts import tabular_metrics
from functools import partial

#openai.api_key = "YOUR_API_KEY"
suite = openml.study.get_suite(271) # Classification
# suite = openml.study.get_suite(269) # Regression
tasks = suite.tasks

for task_id in tasks:
    task = openml.tasks.get_task(task_id)
    this_prompt = create_prompt(task)
    print(this_prompt)
    break
