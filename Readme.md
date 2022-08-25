This is the project prepared for OLG interview round 2. 

## Introduction

The project is about predicting credit risk rating using machine learning algorithms with client data provided, including 118 continuous variables and a few categorical variables such as Country Code and Period. The original data provided is stored in `/data/raw/df_client.csv`. The task has been divided into 5 parts:

- Exploratory Data Analysis (EDA)
- Feature Selection
- Feature Engineering
- Model Fitting and Tuning
- Model Evaluation

Beside manually going through the above processes, an automated procedure using `AutoViML` library is also implemented to have its results compared to the results from the manual processes.

## EDA

In this part, I gained some basic information about the original dataset, such as total number of rows and columns, number of missing data, distribution of the outcome, and relationship between variables. The plots were generated using `matplotlib` and `seaborn`.

## Feature Selection

The steps taken in this part includes:

- Splitting the dataset into 60% training, 20% validation, and 20% testing
- Missing data and outlier handling (4 methods)
    - removing columns with more than 20% missing data, and then list-wise removing
    - removing columns with more than 40% missing data, and then list-wise removing
    - mean imputation, using `SimpleImputer` class in `sklearn`
    - regression imputation, using `RegressionImputer` class in `sklearn`
- Selecting features using correlation between features (unsupervised)
- Selecting features based on their statistical relationships to the predicting outcome

Four sets of training data were obtained using 4 different methods to handle missing data. The 4 datasets were later evaluated using a simple model to decide the optimal one. On each of the 4 datasets, I applied similar processes for feature selection: removing variables with inter-feature correlation of at least 0.7, and then using Kendall's rand coefficient to filter out features that are not significantly impacting the outcome. The four training datasets, together with the selected features and the models for imputation, were then saved for future use in the next section.

## Feature Engineering

In this section, for each of the four training datasets, the continuous features are standardized, followed by one-hot encoding the categorical variables and label encoding the outcome. The four training datasets were then each fitted with a simple `sklearn.tree.DecisionTreeClassifier` model, and evaluated with the validation dataset, treated with the same method as the corresponding training dataset. It turned out that the dataset  handled with regression imputation yielded the best f1 score, which is a popular measurement for multinomial classification. Note that f1 score measures both the precision and the recall:
\[\text{F1 score} = 2 \times \frac{\mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}} \,,\]
where
\[\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}} \,, \quad \text{and} \quad \mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}} \,,\]
with TP, TN, FP, FN representing the number of true positive, the number of true negative, and the number of false positive, and the number of false negative.

Since regression imputation yielded the best results, the training, validation, and testing dataset treated with this method as well as the mentioned standardization and encoding, were saved as the final cleaned datasets ready for use in the model fitting section.

## Model Fitting and Tuning

### Fitting

Several classification models were fitted without much hyperparameter tuning first to roughly filter out models that does not perform very well. The fitted models include:

- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.ensemble.BaggingClassifier`
- `sklearn.linear_model.LogisticRegression`
- `sklearn.neighbors.KNeighborsClassifier`
- `xgboost.XGBClassifier`

The well-performing models are random forest classification, logistic regression, and gradient boosting. I chose to fine tune the random forest model and the gradient boosting model to optimize their performance further.

### Tuning

The model tuning was done with `from sklearn.model_selection.GridSearchCV`, which takes in combinations of ranges of parameters and evaluates the model with each combination of parameters. After tuning, the random forest model improved its f1 score from 0.2131 to 0.2274, and the gradient boosting model improved from 0.2111 to 0.2210. The two tuned models were re-fitted using the combination of the training dataset and the validation dataset.

## Model Evalutation

The two tuned model were tested with the testing dataset. The random forest model yielded an f1 score of 0.2271, and the gradient boosting model yielded 0.2278. Both the two values were very close to that of the validation results, which ensured the generalization of the two models.

## Additional: Automating the process

Beside all of the above procedures, I also used an automated machine learning library, [AutoViML](https://github.com/AutoViML/Auto_ViML), together with its companion, [AutoViz](https://github.com/AutoViML/AutoViz) and [Featurewiz](https://github.com/AutoViML/featurewiz). Only a few lines of code were required to finish all the manual procudures mentioned before. In addition, the automatically fitted model had a better performance.


























