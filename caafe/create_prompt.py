# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:06:58 2023

@author: 20210595
"""
import openml
import os

def create_description_prompt(dataset_opneml):
    head = """\
    The dataframe `df` is loaded and in memory. Columns are also named attributes.
    Description of the dataset in `df` (column dtypes might be inaccurate):
    """


    X, y, categorical_indicator, attribute_names = dataset_opneml.get_data()

    # Get number of rows
    rows_number = f"""\
This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {str(len(X))}

This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting "{dataset_opneml.default_target_attribute}".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore.
    """

    name = f'**{dataset_opneml.name} Database** \n \n'
    general_description = ''.join((dataset_opneml.description,
                                   '\n \n \n',
                                   ))

    # Technical description

    nan_values = list(X.isnull().mean() * 100)
    dtypes_values = X.dtypes
    list_dtypes = [str(element) for element in dtypes_values]
    samples_list = [str(list(X[name_col][:10])) for name_col in list(X.columns)]
    columns_description = "Columns in `df` (true feature dtypes listed here, categoricals encoded as int):\n"""
    columns_description = name + general_description + columns_description

    var_1_example = list(X.columns)[0]
    var_1_list_example = str(list(X[list(X.columns)[0]][:3]))
    var_2_example = list(X.columns)[1]
    var_2_list_example = str(list(X[list(X.columns)[1]][:3]))
    for name_feature, nan_val_percentage, dtypes_val, sample_val in zip(list(X.columns), nan_values, list_dtypes,
                                                                        samples_list):
        this_line = f'{name_feature} ({dtypes_val}): NaN-freq [{nan_val_percentage}%], Samples {sample_val} \n'
        columns_description += this_line

    data_cleaning = f"""\
Data cleaning is an essential step in preparing datasets for machine learning. This code was written by an expert data scientist working to improve predictions by cleaning the dataset. It is a snippet of code that may do one or more or the next procedures:

This code generates a cleaner dataset (if necessary) that is useful for a downstream classification algorithm (such as XGBoost) predicting "{{dataset_opneml.default_target_attribute}}". 
Here are some of the most common procedures and techniques used in data cleaning for machine learning that can be applied: Handling Missing Values, Deleting rows or columns with a high proportion of missing values if they are not critical for analysis, Imputing missing values by replacing them with statistical measures like mean, median, or mode, dealing with Outliers, encoding categorical Variable, removing duplicate records, Handling Skewed or Imbalanced Data, Standardizing or Normalizing Features, Feature Selection.

The classifier will be trained on the resulting cleaned dataset and evaluated on a holdout set. The evaluation metric is accuracy. The best-performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore.

Code formatting for each cleaning step:
```python
# (Procedure name and description)
# Usefulness: (Description why this procedure adds useful real-world knowledge to simplify the classification of "{dataset_opneml.default_target_attribute}" according to dataset description and attributes dtypes.)
```end

Code formatting for a cleaning step, e.g. Encoding Categorical Variables:
```python
# (Procedure name and description)
# Explanation why this step is necessary
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
```end


Code formatting for cleaning step, e.g. replace nan values with mean:
```python
# (Procedure name and description)
# Explanation why this step is necessary 
df.fillna(df.mean())
```end


Each codeblock generates exactly one cleaning step
Each codeblock ends with ```end and starts with "```python"
Codeblock:

"""

    tail = f"""\
Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to classify "{dataset_opneml.default_target_attribute}" according to dataset description and attributes.)
# Input samples: (Three samples of the columns used in the following code, e.g. '{var_1_example}': {var_1_list_example}, '{var_2_example}': {var_2_list_example}, ...)
(Some pandas code using '{var_1_example}', '{var_2_example}', ... to add a new column for each row in df)
```end

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```end

Each codeblock generates exactly one useful column and can drop unused columns (Feature selection).
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""

    return head, columns_description, rows_number, tail, data_cleaning


def main(task):
    folder_name = 'output'
    #Create folder for the outputs

    if not os.path.exists("output"):
        # if the output directory is not present
        # then create it.
        os.makedirs("output")

    datasetID = task.dataset_id
    dataset = openml.datasets.get_dataset(datasetID)
    # Let's build the description_attributes:
    head, description_attributes, rows_number, tail, data_cleaning_output = create_description_prompt(dataset)
    with open(f"{folder_name}/{str(datasetID)}_prompt.txt", 'a', encoding="utf-8") as fh:
        fh.write(head)
        fh.write(f"{description_attributes}\n")
        fh.write(f"{data_cleaning_output}")
        fh.write(f"{rows_number}\n")
        fh.write(tail)

    this_prompt = '\n'.join([head,
                             description_attributes,
                             data_cleaning_output,
                             rows_number,
                             tail])
    return this_prompt

