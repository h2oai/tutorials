 # Introduction to Machine Learning with H2O - Part 1

## Outline

- [Objective](#objective)
- [Pre-requisites](#pre-requisites) 
- [Overview](#overview)
- [Task 1: Import H2O, libraries, and estimators. Initialize H2O and load the dataset](#task-1-import-h2o-libraries-and-estimators-initialize-h2o-and-load-the-dataset)
- [Task 2: Concepts](#task-2-concepts)
- [Task 3: Take a quick look at the data, split the dataset , and select the predictor(s) and response variables](#task-3-take-a-quick-look-at-the-data-split-the-dataset-and-select-the-predictor(s)-and-response-variables)
- [Task 4: Build a GLM with default settings and inspect the results](#task-4-build-a-glm-with-default-settings-and-inspect-the-results)
- [Task 5: Build a Random Forest with default settings and inspect the initial results](#task-5-build-a-random-forest-with-default-settings-and-inspect-the-initial-results)
- [Task 6: Build a GBM with default settings](#task-6-build-a-GBM-with-default-settings)
- [Task 7: Tuning the GLM with H2O GridSearch](#task-7-tuning-the-glm-with-h2o-gridsearch)
- [Task 8: Tuning the RF model with H2O GridSearch](#task-8-tuning-the-rf-model-with-h2o-gridsearch)
- [Task 9: Tuning the GBM model with H2O-GridSearch](#task-9-tuning-the-gbm-model-with-h2o-gridsearch)
- [Task 10: Test set Performance](#task-10-test-set-performance)
- [Task 11: Challenge & Shutting down your Cluster](#task-11-challenge-&-shutting-down-your-cluster)
- [Task 12: Next Steps](#task-12-next-steps)

## Objective
We will be using a subset of the Freddie Mac Single-Family dataset to try to predict whether or not a mortgage loan will be delinquent using H2O’s GLM, Random Forest, and GBM models. We will go over how to use these models for classification problems, and we will demonstrate how to use H2O’s grid search to tune the hyper-parameters of each model.

## Pre-requisites 
Some basic knowledge of machine learning. Familiarity with Python. Make sure you have Jupyter Notebook installed on your local machine, and that you have already installed H2O-3.

If you do not have H2O-3, you can follow the installation guide on the [H2O Documentation page](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html)

We recommend creating an Anaconda Cloud environment, as shown in the installation guide, [Install on Anaconda Could.](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html#install-on-anaconda-cloud) This would guarantee that you will have everything that you need to do this tutorial. 
## Overview
The data set we’re using comes from Freddie Mac and contains 20 years of mortgage history for each loan and contains information about "loan-level credit performance data on a portion of fully amortizing fixed-rate mortgages that Freddie Mac bought between 1999 to 2017. Features include demographic factors, monthly loan performance, credit performance including property disposition, voluntary prepayments, MI Recoveries, non-MI recoveries, expenses, current deferred UPB and due date of last paid installment."[1] 

We’re going to use machine learning with H2O to predict whether or not a loan holder will default. To do this we are going to build three classification models: a Linear model, Random Forest, and a Gradient Boosting Machine model, to predict whether or not a loan will be delinquent. Complete this tutorial to see how we achieved those results.

[1] Our dataset is a subset of the [Freddie Mac Single-Family Loan-Level Dataset.](http://www.freddiemac.com/research/datasets/sf_loanlevel_dataset.html) It contains about 500,000 rows and is about 80 MB.
## Task 1: Import H2O, libraries, and estimators. Initialize H2O and load the dataset
We will start by importing H2O, the estimators for the algorithms that we will use, and also the function to perform Grid Search on those algorithms. 

``` python
#Import H2O and other libraries that will be used in this tutorial 
import h2o
import matplotlib as plt

#Import the Estimators
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator

#Import h2o grid search 
import h2o.grid 
from h2o.grid.grid_search import H2OGridSearch
```
We now have to initialize an H2O cluster or instance, in this case. You can specify how much maximum memory you want your cluster to have. This will only guarantee that the cluster will not use more than 4GB from your machine memory. We will assign 4GB of memory, although we won’t be using all of it. Typically we recommend an H2O Cluster with at least 3-4 times the amount of memory as the dataset size. 

``` python
h2o.init(max_mem_size="4G")
```

![cluster-info](assets/cluster-info.jpg)

 After initializing the H2O cluster, you will see the information shown above. Basically, we have an H2O cluster with just 1 node. Clicking on the link will take you to your **Flow instance** where you can see your models, data frames, plots and much more. Click on the link and it will take you to a window similar to the one below. Keep it open in a separate tab, as we will come back to it later on.

![flow-welcome-page](assets/flow-welcome-page.jpg)

Next, we will import the dataset. You can download H2O's subset of the Freddie Mac Single-Family Loan-Level dataset to your local drive and save it at as csv file. Loan_level_500k.csv. [Loan_Level_500k.csv](https://s3.amazonaws.com/data.h2o.ai/DAI-Tutorials/loan_level_500k.csv)
Make sure that the dataset is in the same directory as your Jupyter Notebook. For example, if your Jupyter file is in your **Documents** save the csv file there. Or, you can just specify the path of where the file is located; in our case, the file is in S3. That’s why we’ll just do the following: 

``` python
#Import the dataset 
loan_level = h2o.import_file("https://s3.amazonaws.com/data.h2o.ai/DAI-Tutorials/loan_level_500k.csv")
```
```
Parse progress: |█████████████████████████████████████████████████████████| 100%
```

Now that we have our dataset, we will explore some concepts and then do some exploration of the data and prepare it for modeling.

## Task 2: Concepts 

### H2O
H2O is an open-source, in-memory, distributed, fast, and scalable machine learning and predictive analytics platform that allows you to build machine learning models on big data and provides easy productionalization of those models in an enterprise environment.

H2O's core code is written in Java. Inside H2O, a Distributed Key/Value store is used to access and reference data, models, objects, etc. across all nodes and machines. The algorithms are implemented on top of H2O's distributed Map/Reduce framework and utilize the Java Fork/Join framework for multi-threading. The data is read in parallel and is distributed across the cluster and stored in memory in a columnar format in a compressed way. H2Oâ€™s data parser has built-in intelligence to guess the schema of the incoming dataset and supports data ingest from multiple sources in various formats.

The speed, quality, ease-of-use, and model-deployment for the various cutting edge Supervised and Unsupervised algorithms like Deep Learning, Tree Ensembles, and GLRM make H2O a highly sought after API for big data data science.

### Flow

H2O Flow is an open-source user interface for H2O. It is a web-based interactive environment that allows you to combine code execution, text, mathematics, plots, and rich media in a single document.

With H2O Flow, you can capture, rerun, annotate, present, and share your workflow. H2O Flow allows you to use H2O interactively to import files, build models, and iteratively improve them. Based on your models, you can make predictions and add rich text to create vignettes of your work - all within Flow’s browser-based environment.

Flow’s hybrid user interface seamlessly blends command-line computing with a modern graphical user interface. However, rather than displaying output as plain text, Flow provides a point-and-click user interface for every H2O operation. It allows you to access any H2O object in the form of well-organized tabular data.

H2O Flow sends commands to H2O as a sequence of executable cells. The cells can be modified, rearranged, or saved to a library. Each cell contains an input field that allows you to enter commands, define functions, call other functions and access other cells or objects on the page. When you execute the cell, the output is a graphical object, which can be inspected to view additional details.

### Supervised Learning 
Supervised learning is when the dataset contains the output that you are trying to predict and you use an algorithm to try to predict a function between your input and output, such as 
y=f(X)
With supervised learning, you train your algorithms to try to approximate a function that will allow you to predict the variable y.

### Binary Classifier
A binary classification model predicts in what two categories(classes) the elements of a given set belong to. In the case of our example, the two categories(classes) are defaulting on your home loan and not defaulting. 

Binary classifications produce four outcomes:

**Predicticted as Positive:**
True Positive = TP = Actual Positive labels predicted as positives
False Positive = FP = Actual Negative labels predicted as positives
 
**Predicted as Negative:**
True Negative = TN = Actual Negative labels predicted as negatives
False Negative = FN = Actual Positive labels predicted as negatives


### Confusion Matrix
The confusion matrix is also known as the error matrix since it makes it easy to visualize the classification rate of the model including the error rate. With the confusion matrix you can see the frequency with which a machine learning model confuses one label with another, and thus the name “confusion matrix”.

### ROC
An essential tool for classification problems is the ROC Curve or Receiver Operating Characteristics Curve. The ROC Curve visually shows the performance of a binary classifier; in other words, it "tells how much a model is capable of distinguishing between classes" [1] and the corresponding threshold. 
The ROC curve plots the sensitivity or true positive rate (y-axis) versus the Specificity or false positive rate (x-axis) for every possible classification threshold. A classification threshold or decision threshold is the probability value that the model will use to determine where a class belongs. The threshold acts as a boundary between classes to determine one class from another. Since we are dealing with probabilities of values between 0 and 1, an example of a threshold can be 0.5. This tells the model that anything below 0.5 is part of one class and anything above 0.5 belongs to a different class.
A ROC Curve is also able to tell you how well your model did by quantifying its performance. The scoring is determined by the percent of the area that is under the ROC curve otherwise known as **Area Under the Curve or AUC.** The closer the ROC curve is to the left (the bigger the AUC percentage), the better the model is at separating between classes.

The ROC curve is a useful tool because it only focuses on how well the model was able to distinguish between classes. "AUC's can help represent the probability that the classifier will rank a randomly selected positive observation higher than a randomly selected negative observation" [2]. However, on rare occurrences, a high AUC could provide a false sense that the model is correctly predicting the results

### Precision and Recall
**Precision** is the ratio of correct positive predictions divided by the total number of positive predictions. This ratio is also known as positive predictive value and is measured from 0.0 to 1.0, where 0.0 is the worst and 1.0 is the best precision.

**Recall** is the true positive rate which is the ratio of the number of true positive predictions divided by all positive true predictions. Recall is a metric of the actual positive predictions. It tells us how many correct positive results occurred from all the positive samples available during the test of the model.

### F1 Score
The F1 Score is another measurement of classification accuracy. It represents the harmonic average of precision and recall. F1 is measured in the range of 0 to 1, where 0 means that there are no true positives, and 1 when there is neither false negatives or false positives or perfect precision and recall[3].


### Accuracy
Accuracy or ACC (not to be confused with AUC or area under the curve) is a single metric in binary classification problems. ACC is the ratio of the number of correct predictions divided by the total number of predictions. In other words, it describes how well the model can correctly identify both the true positives and true negatives. Accuracy is measured in the range of 0 to 1, where 1 is perfect accuracy or perfect classification, and 0 is poor accuracy or poor classification[4].
Using the confusion matrix table, ACC can be calculated in the following manner:
Accuracy = (TP + TN) / (TP + TN + FP + FN)

### Log loss
Log loss also measures the performance of classification models with a focus on the uncertainty between predictions and actual results. The log loss score is the average log-loss across all observations. Log loss is measured in the range of 0 to 1, where a model with a log loss of 0 would be the perfect classifier and 1 the worst. When the model is unable to make correct predictions, the log loss increases making the model a poor model[5].

### Cross-Validation
Cross validation is a model validation technique in which you can check how well a statistical analysis or model, will perform on an independent dataset. We use cross validation to see how our model will predict on unseen data. We will explore two cross validation approaches. The first one is to take the training data and split it into training and validation set, which is similar to a test set. This approach is called **validation set cross validation.** Also, there is **K-Fold** cross validation, in which you do not need to split the data, but use the entire dataset. Depending on the number of *folds* that you choose during training, the data is divided into k groups; k-1 groups are trained, and then, the last group serves to evaluate. After every group has been used to evaluate the model, the average of all the scores is obtained; and thus, we obtain a validation score[6].

[1] [Towards Data Science - Understanding AUC- ROC Curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)
 
[2] [ROC Curves and Under the Curve (AUC) Explained](https://www.youtube.com/watch?v=OAl6eAyP-yo)
 
[3] [Wiki F1 Score](https://en.wikipedia.org/wiki/F1_score)
 
[4] [Wiki Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision)
 
[5] [Wiki Log Loss](http://wiki.fast.ai/index.php/Log_Loss)
 
[6] [Towards Data Science - Cross-Validation](https://towardsdatascience.com/cross-validation-70289113a072)

## Task 3: Take a quick look at the data, split the dataset, and select the predictor(s) and response variables

To ensure the dataset was properly imported use the `.head()` command 

```
loan_level.head()
```
![dataset-head](assets/dataset-head.jpg)


Your results should look like the table above. If you scroll to the right, you will be able to see all the features in our dataset. 
We can also take a look at a quick statistical summary of our dataset with the `.describe()` command as shown below

``` python
loan_level.describe()
```

![data-describe](assets/data-describe.jpg)

The total number of rows in our dataset is 500,137 and the total number of features or columns is  27. Additionally, you will get a sense of the spread of each of your columns, the column type, as well as the number of missing and zero values in your dataset.

Let's take a quick look at the response column by checking the distribution

``` python
loan_level["DELINQUENT"].table()
```

![response-distribution](assets/response-distribution.jpg)

As you can see, we have a very imbalanced dataset, as only 3.6% of the samples are TRUE labels, meaning that only 3.6% of the samples in the dataset have been labeled as `DELINQUENT.`

You can also do the same thing with H2O Flow, by clicking ‘import’ and then viewing the actual table once it’s imported. Go to your Flow instance and add a new cell

![flow-add-cell](assets/flow-add-cell.jpg)

Copy and paste the following line of code in the new cell and run it. Then, click on **Parse these files** 

```
importFiles ["https://s3.amazonaws.com/data.h2o.ai/DAI-Tutorials/loan_level_500k.csv"]
```

![flow-parse-file](assets/flow-parse-file.jpg)

After clicking on **Parse these files,** you will see a parse set-up similar to the image below

![flow-parse-set-up](assets/flow-parse-set-up.jpg)



H2O will try to parse the file and assign appropriate column types. But you can change column types if they’re not imported correctly. After you have inspected the parse set-up, click on parse. 

Once finished, you will see the following message, confirming that the parsing was completed. 

![flow-parse-finished](assets/flow-parse-finished.jpg)

**The main goal of this tutorial is to show the usage of some models for classification problems, as well as to tune some of the hyperparameters of the models. For that reason, we will be skipping any data visualization, and manipulation, as well as feature engineering. The aforementioned stages in machine learning are very important, and should always be done; however, they will be covered in later tutorials.**

Since we have a large enough dataset, we will split our dataset into three sets and we will call them **train, valid,** and **test.** We will treat the test set as if it is some unseen data in which we want to make predictions, and we will use the valid set for validation purposes and to tune all our models. We will not use the test set until the end of the tutorial to check the final scores of our models. 

Return to your Jupyter Notebook to split our dataset into three sets. We will use the `.split_frame () command.` Note that we can do this in one line of code. Inside the split function, we declare the ratio of the data that we want in our first set, in this case, **train** set. We will assign 70% to the training set, and 15% for the validation, as well as for the test set. The random seed is set to 42 just for reproducibility purposes. You can choose any random seed that you want, but if you want to see the consistent results, you will have to use the same random seed anytime you re-run your script. 

``` python
train, valid, test = loan_level.split_frame([0.7, 0.15], seed=42)
```
We can check the distribution of the data split by checking the number of rows in each set.

```python
print("train:%d valid:%d test:%d" % (train.nrows, valid.nrows, test.nrows))
train:350268 valid:74971 test:74898
``` 
Now we will split the dataset in our Flow instance. Click on **View,** then **Split** and use the default ratios of 0.75 for **train,** and 0.25 for **test** and change the names accordingly. Also, change the **seed** to 42 and click **Create** 

![flow-split-data](assets/flow-split-data.gif)

Next, we need to choose our **predictors**, or **x variable**, and our **response** or **y variable**. For the H2O-3 estimators, we do not use the actual data frame, but strings containing the name of the columns in our dataset.

Return to your Jupyter Notebook; for our y variable, we will choose `DELINQUENT` because we want to predict whether or not a loan will default. For the x variable, we will choose all but 4 features. One is the feature that we will predict, and then `PREPAID` and `PREPAYMENT_PENALTY_MORTGAGE_FLAG` because they are clear indicators if a loan is or is not delinquent and we will not have the information at the time deciding whether to give a loan or not. In machine learning terms, introducing these types of features is called leakage. And lastly, `PRODUCT_TYPE` because that’s a constant value for every row, meaning all samples have the same value; therefore, this feature will not have any predictive value.

There are several ways to choose your predictors, but for this tutorial, just substract the columns in `ignored` from the names in the training set. 

``` python
y = "DELINQUENT"

ignore = ["DELINQUENT", "PREPAID", "PREPAYMENT_PENALTY_MORTGAGE_FLAG", "PRODUCT_TYPE"] 

x = list(set(train.names) - set(ignore))
```

If you want to see the list of the features that are in your x variable, just print x.

``` python
print(x)
```
```
['CREDIT_SCORE', 'FIRST_PAYMENT_DATE', 'FIRST_TIME_HOMEBUYER_FLAG', 'MATURITY_DATE', 'METROPOLITAN_STATISTICAL_AREA', 'MORTGAGE_INSURANCE_PERCENTAGE', 'NUMBER_OF_UNITS', 'OCCUPANCY_STATUS', 'ORIGINAL_COMBINED_LOAN_TO_VALUE', 'ORIGINAL_DEBT_TO_INCOME_RATIO', 'ORIGINAL_UPB', 'ORIGINAL_LOAN_TO_VALUE', 'ORIGINAL_INTEREST_RATE', 'CHANNEL', 'PROPERTY_STATE', 'PROPERTY_TYPE', 'POSTAL_CODE', 'LOAN_SEQUENCE_NUMBER', 'LOAN_PURPOSE', 'ORIGINAL_LOAN_TERM', 'NUMBER_OF_BORROWERS', 'SELLER_NAME', 'SERVICER_NAME']
``` 
## Task 4: Build a GLM with default settings and inspect the results

Now that we have our train, valid, and test sets, as well as our x and y variables defined, we can start building models! We will start with an H2O Generalized Linear Model (GLM). A GLM fits a generalized linear model, specified by a response variable, a set of predictors, and a description of the error distribution. Since we have a binomial classification problem, we have to specify the family, in this case, it will be “binomial.” 

Since we already imported the H2O GLM estimator, we will just instantiate our model. For simplicity, the name of our model will be `glm`. To build a GLM, you just need to define the family and you are ready to go. However, we will set a random seed for reproducibility purposes, and also a model id to be able to retrieve the model later on if we need to access it. You can instantiate your GLM as shown below. 

``` python
glm = H2OGeneralizedLinearEstimator(family = "binomial", seed=42, model_id = 'default_glm')
```
Now we will train our GLM model. To do so, we just use the `.train()` function. In the train function, we need to specify the predictors (x), the response (y), the training set (train) and a validation_frame, if you have one. In our case, we have our valid set, which we will use. 

``` python
%time glm.train(x = x, y = y, training_frame = train, validation_frame = valid)
```
**Note:** The `%time` in front of our train command is used to display the time it takes to train the model, and it’s a feature from Jupyter Notebook, and it does not work on a command line or outside of Jupyter Notebook. 

You can do the same thing in Flow with the ‘Build model’ dialog. Click on your train set, and click on **Build Model,** then scroll down to the “Build a Model” cell, and select **Generalized Linear Modeling** for the algorithm. For model id, you can use ‘flow_default_glm.’ Instead of doing cross validation with a validation set, we are going to use Flow’s K-fold cross validation; therefore, type **5** for ‘nfolds,’ and set the random seed to 42. Again, choose "DELINQUENT" for your ‘response_column’ and for the ignored columns, choose  "PREPAYMENT_PENALTY_MORTGAGE_FLAG," "PRODUCT_TYPE," "PREPAID." Lastly, choose **binomial** for ‘family’

![flow-default-glm](assets/flow-default-glm.gif)

You have now built and trained a GLM! If you type the name of your model in a new cell and run it, H2O will give you a complete summary of your model. You will see your model’s metrics on the training and validation set. From the model details, you will also see a short summary with the parameters of your model, the metrics of your model, the confusion matrix, maximum matrices at different thresholds, a Gains/Lift table, and the scoring history. (Gains/Lift and scoring history are not shown)

![default-glm-details-1](assets/default-glm-details-1.jpg)


![default-glm-details-2](assets/default-glm-details-2.jpg)

From the summary results, we can see the GLM performance. We will focus on the Area Under the Curve (AUC), and since we have a very imbalanced dataset, we will be looking at the F1 score. Additionally, we will also take a quick look at the misclassification error and logloss. 

From the report, we can look at the metrics on the training and validation data, and we see that the training AUC was 0.8504 while the validation AUC was 0.8451

![default-glm-training-metrics](assets/default-glm-training-metrics.jpg)


![default-glm-validation-metrics](assets/default-glm-validation-metrics.jpg)

From the report, we can also see the max F1 score. For the default GLM, we obtained a training F1 score of **0.2888** and a validation F1 score of **0.2835.** 

**Training maximum metrics**

![default-glm-training-max-metrics](assets/default-glm-training-max-metrics.jpg)

**Validation maximum metrics**

![default-glm-validation-max-metrics](assets/default-glm-validation-max-metrics.jpg)

We can plot the Scoring history for any of our models as shown below

``` python
glm.plot()
```
![default-glm-scoring-history](assets/default-glm-scoring-history.jpg)


We can see from the plot above that after 3 iterations, the score does not keep improving; therefore, if we need to set a number of iterations as a future parameter we can choose 3, as the scores don’t really improve after that point. We can also use the default number of iterations and use early stopping, that way the model will stop training when the model is no longer improving. We will use early stopping when we start tuning our models.

We can also plot a variable importance plot, to see how each of our features contribute to the linear model. 

```
glm.varimp_plot()
```

![default-glm-var-imp](assets/default-glm-var-imp.jpg)


From the variable importance plot, we can see that the most significant feature is `SERVICER_NAME.` In the most important feature, we have different banks or “servicers” and in our linear model, each one makes a difference; for that reason, we see that the first 4 variables in the plot above are 4 of the servicers in the dataset. These services are the most influential to our model in making predictions of whether someone will default or not. Please keep in mind that it does not necessarily mean that if someone gets a loan from Wells Fargo they will have a high probability of default.

We will take a look at the first ten predictions of our model with the following command:
``` python
glm.predict(valid)
``` 
 
![default-glm-predictions](assets/default-glm-predictions.jpg)

**Note:** if you want to see the more predictions use the `.as_data_frame()` after the above line of code, and that should allow you to view all the predictions on the validation set. The line of code should look something like the following,
```python
glm.predict(valid).as_data_frame()
```
The model used by H2O for this classification problem is a Logistic Regression model and the predictions are based on the threshold for each probability[1]. For a binary classifier, H2O predicts the labels based on the maximum F1 threshold. From the report, the threshold for max F1 is  0.129. So, any time the probability for TRUE is greater than the 0.129, the predicted label will be TRUE. As is in the case of the sixth prediction. To learn more about predictions, you can visit the [Prediction Section](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/performance-and-prediction.html?highlight=predict#prediction) from the H2O documentation.

Lastly, save the default performance of the model, as we will use this for comparison purposes later on.
```python
default_glm_perf=glm.model_performance(valid)
```

Go back to the Flow instance, go to the model you created earlier on and click on “View”

![flow-view-default-glm](assets/flow-view-default-glm.jpg)



Expand the Model parameters tab, and you will see a description of the parameters for your model. 

![flow-default-glm-params](assets/flow-default-glm-params.jpg)


If you scroll down, you will see some plots derived from the training data. The first one is the scoring history plot. 

![flow-default-glm-scoring-history](assets/flow-default-glm-scoring-history.jpg)

We can see that the scoring history from Flow shows us that after 3 iterations the score does not improve. Even though we are doing different cross validation, validation set approach in the python script, and k-fold cross validation in Flow, we obtained the same results, indicating that 3 iterations are enough. 

If you continue scrolling down, you will see:
ROC Curve  Training Metrics
ROC Curve  Cross Validation Metrics
Standardized Coefficients Magnitudes 
Training Metrics - Gains/Lift Table
Cross Validation Metrics - Gains/Lift Table
And then all types of outputs 
When all the tabs are collapsed, you will see the following list of details from your model:
![flow-default-glm-details](assets/flow-default-glm-details.jpg)



You can also take a look at the details of the model you built in your Jupyter Notebook. Scroll up to the “Assist” Cell and click on **getModels**

![flow-get-models](assets/flow-get-models.jpg)


Then select ‘default_glm’

![flow-list-of-models](assets/flow-list-of-models.jpg)


You can inspect all the plots and outputs from your model in Flow. The plots that we generated in the Jupyter Notebook are automatically created in Flow, so if you prefer, you can just create your model in Jupyter and then analyze the results in Flow. 

[1] [H2O-3 GLM Logistic Regression](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html#logistic-regression-binomial-family)

## Task 5: Build a Random Forest with default settings and inspect the initial results
We will build a default Distributed Random Forest (DRF) model and see how it performs on our validation set. DRF generates a forest of classification or regression trees, rather than a single classification or regression tree. Each of these trees is a weak learner built on a subset of rows and columns. More trees will reduce the variance. Both classification and regression take the average prediction over all of their trees to make a final prediction, whether predicting for a class or numeric value. 

To build and train our Random Forest or RF(as we will be referring to from this point on) model, simply run the following two lines of code:
``` python
rf = H2ORandomForestEstimator (seed=42, model_id='default_random_forest')
%time rf.train(x=x, y=y, training_frame=train, validation_frame=valid)
```
Note that we defined the random seed and the model id. You do not need to do this, the model can be built without defining these parameters. The reason for choosing the random seed is for reproducibility purposes, and the model id is to easily recognize the model in Flow. 

Again, print the summary of your model as we did with the GLM model. You will see the summary of the model with the default settings, and the metrics score on the training and validation data. 

Below you will see some of the details from the model we just built. 

The AUC and F1 Score reported on the training data are **0.8033** and  **0.2621** respectively, and you can see them in the image below.

![default-rf-training-metrics](assets/default-rf-training-metrics.jpg)

**Results from validation data**

The AUC and F1 Score reported on the validation data are **0.8264** and  **0.2828** respectively.

![default-rf-validation-metrics](assets/default-rf-validation-metrics.jpg)


Let’s build an RF model in Flow. Scroll up again to the Assist” Cell, and click on **buildModel** 

![flow-build-model](assets/flow-build-model.jpg)

In the select algorithm option, choose **Distributed Random Forest,** then change the model id to `flow_default_rf.` Click on the *training_frame* option and select **train.**  Change *nfolds* so that it is 5. Choose "DELINQUENT" for your *response_column* and for the ignored columns, choose  "PREPAYMENT_PENALTY_MORTGAGE_FLAG," "PRODUCT_TYPE," "PREPAID."
![flow-default-rf](assets/flow-default-rf.gif)

If you would like, you can view the outputs of your RF model in Flow. However, we can also generate the plots in our Jupyter Notebook.

``` python
rf.plot(metric='auc')
```
You will see a plot similar to the one below

![default-rf-scoring-history](assets/default-rf-scoring-history.jpg)

In this case, we see that the RF model is far from overfitting because the training error is still lower than the validation error and that means that we can probably do some tuning to improve our model. 

We can also generate the variable importance plot,
```python
rf.varimp_plot(20)
```

![default-rf-var-imp](assets/default-rf-var-imp.jpg)

It is interesting to see that for our RF model, `PROPERTY_STATE` Is the most important variable, implying that the prediction of whether a loan could be delinquent or not depends on the state where someone is trying to buy that property. The second most important is a more intuitive one, which is the CREDIT_SCORE, as one could expect someone with really good credit to fully pay their loans. 

If you want to check the options of what you can print from your model, just type the name of your model along with a dot (“.”) and press tab. You should see a drop-down menu like the one shown in the image below. 

![printing-options](assets/printing-options.jpg)

Keep in mind that for some of them you will need to open and close parentheses at the end in order to display what you want. Let’s say we wanted to print the training accuracy of our model, you could select accuracy, but you need to add parentheses in order to get just the accuracy, otherwise, you will get the entire report again.

![default-rf-acc](assets/default-rf-acc.jpg)

The first parameter shown in the list above is the threshold, and the second value is the accuracy. 

To print the F1 Score you will simply need to type the following line of code,

```python
rf.F1()
```
You will see the output in a list format. First, you will see the threshold, and then the actual value; same as in the accuracy.

Let’s take a look at the first ten predictions in our validation set, and compare it to our first model. 

``` python
rf.predict(valid)
```

![default-rf-pred](assets/default-rf-pred.jpg)

Both models, GLM and RF made the same predictions in the first ten predictions. For e.g. Even the TRUE prediction, for the sixth row is the same; there is a different probability, but the prediction is the same. 

Again, save the model performance on the validation data
```python
rf_default_per = rf.model_performance(valid)
```
## Task 6: Build a GBM with default settings

Gradient Boosting Machine (for Regression and Classification) is a forward learning ensemble method. H2O’s GBM sequentially builds classification trees on all the features of the dataset in a fully distributed way - each tree is built in parallel. H2O’s GBM fits consecutive trees where each solves for the net loss of the prior trees. 
Sometimes GBMs tend to be the best possible models because they are robust and directly optimize the cost function. On the other hand, they tend to overfit, so you need to find the proper stopping point; they are sensitive to noise, and they have several hyper-parameters.

Defining a GBM model is as simple as the other models we have been working with. 
``` python
gbm= H2OGradientBoostingEstimator(seed=42, model_id='default_gbm')
%time gbm.train(x=x, y=y, training_frame=train, validation_frame = valid)

``` 
Print the model summary

**Training metrics:**

![default-gbm-training-metrics](assets/default-gbm-training-metrics.jpg)


**Validation metrics:**

![default-gbm-validation-metrics](assets/default-gbm-validation-metrics.jpg)


Now, we will explore this model in Flow. Go to your Flow instance, click on **getModels** and click on “default_glm”

![get-default-gbm](assets/get-default-gbm.jpg)

Now scroll down to the scoring history and you should see a plot like the one below

![default-gbm-scoring-history](assets/default-gbm-scoring-history.jpg)

From the scoring history, we can see that we can still increase the number of trees, a little bit more, because the validation score is still improving. We will get into more details during the GBM tuning section. 
Scroll down to the variable importance plot, and take a look at it. Notice how the most important variable is `CREDIT_SCORE` for the GBM. If you recall, for RF, `CREDIT_SCORE` was the second most important variable. And the most important variable for RF is the third most important for the GBM. 

![default-gbm-var-imp](assets/default-gbm-var-imp.jpg)

You can continue exploring the results of your GBM model, or go back to the Jupyter Notebook where we will continue. 

The default GBM model had a slightly better performance than the default RF. 
We will do the predictions with the GBM model as well, as we did with the other two models. 

``` python
gbm.predict(valid)
``` 
![default-gbm-preds](assets/default-gbm-preds.jpg)

All three models made the same 10 predictions and this gives us an indication of why all three scores are close to each other. Although the sixth prediction is TRUE for all three models, the probability is not exactly the same, but since the thresholds for all three models were low, the predictions were still TRUE. 
As we did with the other two models, save the model performance.

``` python
default_gbm_per = gbm.model_performance(valid)
```
Next, we will tune our models and see if we can achieve better performance. 
## Task 7: Tuning the GLM with H2O GridSearch 
H2O supports two types of grid search – traditional (or “cartesian”) grid search and random grid search. In a cartesian grid search, you specify a set of values for each hyperparameter that you want to search over, and H2O will train a model for every combination of the hyperparameter values. This means that if you have three hyperparameters and you specify 5, 10 and 2 values for each, your grid will contain a total of 5*10*2 = 100 models.

In a random grid search, you specify the hyperparameter space in the exact same way, except H2O will sample uniformly from the set of all possible hyperparameter value combinations. In the random grid search, you also specify a stopping criterion, which controls when the random grid search is completed. You can tell the random grid search to stop by specifying a maximum number of models or the maximum number of seconds allowed for the search. You can also specify a performance-metric-based stopping criterion, which will stop the random grid search when the performance stops improving by a specified amount.
Once the grid search is complete, you can query the grid object and sort the models by a particular performance metric (for example, “AUC”). All models are stored in the H2O cluster and are accessible by model id. 

To save some time, we will do a random grid search for our GLM model instead of the cartesian search. The H2OGridSearch has **4 parameters,** and in order to use it, you need **at least three** of them. The first parameter for the grid search is the **model** that you want to tune. Next are your **hyperparameters,** which needs to be a string of parameters, and a list of values to be explored by grid search. The third one is optional, which is the **grid id,** and if you do not specify one, an id will automatically be generated. Lastly, the **search criteria,** where you can specify if you want to do a cartesian or random search.  

We will explore two ways of defining your grid search and you can use the way you prefer. One way is to define all at once in the grid search (as we will do it for the GLM). The second way is to define every parameter separately. For example, define your model, your hyper-parameters, and your search criteria, and just add that to your grid search once you are ready.

For our GLM, we can only tune **alpha** and **lambda.** The other parameters that you could change, such as *solver,* *max_active_predictors,* and *nlambdas* to mention a few, are not supported by H2OGridSearch. 

**Alpha** is the distribution of regularization between the L1 (Lasso) and L2 (Ridge) penalties. A value of 1 for alpha represents Lasso regression, a value of 0 produces Ridge regression, and anything in between specifies the amount of mixing between the two. **Lambda,** on the other hand, is the regularization strength. For alpha, we can explore the range from 0 to 1 in steps of 0.01. For lambda, you could start just doing your own random searches, but that might take a lot of time. Instead, we can base our value for lambda on the original value of lambda, which was 6.626e-5. We can choose our starting point to be 1e-6, and go from there. The example of this is shown below. 

``` python
glm_grid = h2o.grid.H2OGridSearch (
    H2OGeneralizedLinearEstimator( 
        family = "binomial",
        lambda_search = True),
    
    hyper_params = {
        "alpha": [x*0.01 for x in range(0, 100)],
        "lambda": [x*1e-6 for x in range(0, 1000)],
        },
    
    grid_id = "glm_grid",
    
    search_criteria = {
        "strategy":"RandomDiscrete",
        "max_models":100,
        "max_runtime_secs":300,
        "seed":42
        }
)
%time glm_grid.train(x=x, y=y, training_frame=train, validation_frame = valid)

        }
    )

%time glm_grid.train(x=x, y=y, training_frame=train, validation_frame = valid)
```
 You can easily see all **four parameters** for our grid search in the code sample above. We defined our GLM model the same way we did before. Then, we take care of the hyper-parameters and notice that we have used a for loop for the ranges of both alpha and lambda in order to cover more possible values. Because the number of possible models is really big, in our search criteria, we specify that we want a maximum number of 100 models, or that the grid search runs for only 300 seconds.
 
Now we will print the models in descending order, sorted by the AUC. By default, the grid will return the best models based on the `logloss`. Therefore, in order to get the best model based on the AUC, we will specify that we want to sort the models by AUC. You can change this to other metrics, depending on what you are looking for.

``` python
sorted_glm_grid = glm_grid.get_grid(sort_by='auc',decreasing=True)
sorted_glm_grid
```
With the code sample above, you will get the models that were created, with their respective alpha, lambda, model id, and AUC. 

Next, we will do the same in Flow. Using a grid search in Flow is as easy as just clicking some boxes and adding some numbers. Go to the “Assist” Cell again, and click on **buildModel.** and select **Generalized Linear Modeling** for the algorithm. Repeat the exact same process as before, when you built the default GLM model. For model id, just use ‘glm.’ type **5** for ‘nfolds,’ and set the random seed to 42. Again, choose "DELINQUENT" for your ‘response_column’ and for the ignored columns, choose  "PREPAYMENT_PENALTY_MORTGAGE_FLAG," "PRODUCT_TYPE," "PREPAID." Lastly, choose **binomial** for ‘family.’ But this time don’t click on **Build Model** yet.

Now, every time you build a model you are given the option to select the grid option, as shown in the image below. For the parameters shown in the image below, just leave them how they are. 

![flow-glm-grid](assets/flow-glm-grid.jpg)

Scroll down to the *alpha* and *lambda* parameters and check the boxes next to them and add the list of numbers shown in the image below. Also, check the *lambda_search* and *standardize* boxes. 

![flow-glm-grid-alpha](assets/flow-glm-grid-alpha.jpg) 

Lastly, for the **Grid Settings** make sure your settings look similar to the ones in the image below. You will need to change the **Grid ID, Strategy, Max Models, Max Runtime and Stopping Metric.** 

![flow-glm-grid-settings](assets/flow-glm-grid-settings.jpg)

Once you have updated the settings, click on **Build Model.** When the model is done, click on **View** and you will see the list of your models. The top model will be the model with the best AUC score. Click on it, and explore the results. Our best model yielded to a validation AUC score of 0.8548, and our ROC curve is shown below. 

![flow-glm-grid-AUC](assets/flow-glm-grid-AUC.jpg)

After looking at the grid search from Flow, let's explore the best model obtained from our grid search. Save the best model, and print the model summary with the following code:

``` python
best_glm_model = glm_grid.models[0] 
best_glm_model.summary()
``` 
With the first line of code, we are retrieving the best model from the grid search, and the second line of code will print the parameters used for the best model found by the grid search. We will do a quick comparison between the performance of the default glm model and the best model from the grid search. 

First, evaluate the model performance on the validation set.

```python
tuned_glm_perf = best_glm_model.model_performance(valid)
``` 
Now, print the AUC for the default, and the tuned model.

``` python
print("Default GLM AUC: %.4f \nTuned GLM AUC:%.4f" % (default_glm_perf.auc(), tuned_glm_perf.auc()))
```
Output:
``` 
Default GLM AUC: 0.8451 
Tuned GLM AUC:0.8460
``` 

The AUC did not really improve. Statistically, it would not be considered an improvement, but it slightly changed. We did not expect the GLM model to perform great, or to have a great improvement with the grid search, as it is just a linear model, and in order to perform well, we would need a linear distribution of our data and response variable. 

We can also print the F1 Score to see if it improved or not,
``` python
print ("Default GLM F1 Score:", default_glm_perf.F1())
print ("Tuned GLM F1 Score", tuned_glm_perf.F1())
```

Output:
``` python
Default GLM F1 Score: [[0.12608455078990774, 0.283510936623668]]
Tuned GLM F1 Score [[0.1308277409434515, 0.2839452843772498]]
```
The max F1 Score did not have a significant improvement. Although the threshold slightly increased, it did not improve the overall F1 Score by much. Let’s take a look at the confusion matrix to see if the values changed.  


``` python
print ("Default GLM: ", default_glm_perf.confusion_matrix())
print ("Tuned GLM: ",  tuned_glm_perf.confusion_matrix())
``` 

![glm-conf-matrix-def-tuned](assets/glm-conf-matrix-def-tuned.jpg)


Notice how the overall error slightly decreased, as well as the error for the FALSE class that was correctly classified. But the error for the TRUE class went up, meaning the model is classifying more samples that are actually TRUE as FALSE. We see that our model has a hard time classifying the TRUE labels, and this is due to the highly imbalanced dataset that we are working on. 

We will do the test evaluation after we tune our other two models.
## Task 8: Tuning the RF model with H2O GridSearch 

We will do the grid search a bit differently this time. We are going to define each parameter of the grid search separately, and then add it to the grid search.

We will first find the two most important parameters for a RF, the maximum depth and then the number of trees. First, we will start with the maximum depth.
**max_depth** defines the number of nodes along the longest path from the start of the tree to the farthest leaf node. Higher values will make the model more complex and can lead to overfitting. Setting this value to 0 specifies no limit. This value defaults to 20. We will first look for the best value for the **max_depth,** this would save us some computational time when we tune the other parameters. As we mentioned before, we will use a slightly different approach for the grid search. We are going to instantiate each parameter for the grid search, and then pass each one into it.

``` python
hyper_parameters = {'max_depth':[1,3,5,6,7,8,9,10,12,13,15,20,25,35]}

rf = H2ORandomForestEstimator(
    seed=42,
    stopping_rounds=5, 
    stopping_tolerance=1e-4, 
    stopping_metric="auc",
    model_id = 'grid_rf'
    )

grid_id = 'depth_grid'

search_criteria = {'strategy': "Cartesian"}

#Grid Search
rf_grid = H2OGridSearch(model=rf, hyper_params=hyper_parameters, grid_id=grid_id, search_criteria=search_criteria)

%time rf_grid.train(x=x, y=y, training_frame=train, validation_frame = valid)

```
After it is done training, print the models sorted by AUC.

``` python
sorted_rf_depth = rf_grid.get_grid(sort_by='auc',decreasing=True)
sorted_rf_depth
```
Now that we have the proper depth for our RF, we will try to tune the next parameter

The other most important parameter that we can tune is the number of trees (`ntrees`). **ntrees** specifies the number of trees that you want your RF to have. When tuning the number of trees, you need to be careful because when you have too many trees, your model will tend to overfit. That’s why it’s always advised to do cross validation, and never tune models based on training scores. Again, you can also use early stopping, that way your model stops training once the validation score is no longer improving.  Let's take a look.

We will use the grid search to build models with 10, 50, 70, 100, 300, 500 and 1000 trees. 

``` python
hyper_parameters = {'ntrees' : [10, 50, 70, 100, 300, 500, 1000]}

rf = H2ORandomForestEstimator(max_depth=12,
    seed=42,
    stopping_rounds=5, 
    stopping_tolerance=1e-4, 
    stopping_metric="auc",
    model_id='rf_ntrees_grid'
    )
grid_id = 'ntrees_grid'
search_criteria = {'strategy': "Cartesian"}

rf_grid = H2OGridSearch(model=rf, 
                        hyper_params=hyper_parameters, 
                        grid_id=grid_id, 
                        search_criteria=search_criteria)

%time rf_grid.train(x=x, y=y, training_frame=train, validation_frame = valid)
```

``` python
sorted_rf_ntrees = rf_grid.get_grid(sort_by='auc',decreasing=True)
sorted_rf_ntrees
```
``` 
   ntrees            model_ids                 auc
0     1000  ntrees_grid_model_7  0.8534127807218633
1      500  ntrees_grid_model_6  0.8530448172815991
2      300  ntrees_grid_model_5  0.8527833608800601
3      100  ntrees_grid_model_4  0.8507529194135223
4       70  ntrees_grid_model_3  0.8497099500887461
5       50  ntrees_grid_model_2  0.8476002325065071
6       10  ntrees_grid_model_1  0.8352835830092651
```
By looking at the training scores, we see that the best model would be the one with 1000 trees! But if you pay close attention, from 300 to 1000 trees it doesn’t really improve much, but the model with 1000 trees takes longer to train than the model with 300 trees.One thing to note here is that the early stopping didn’t come into action because the stopping tolerance is 1e-4, if we were to try we 1e-3, the model probably would’ve stopped earlier. 
Let’s take a look at the scoring history of the best model to see the performance and the difference of using a different number of trees.


``` python
best_ntrees_rf_model = rf_grid.models[0] 
```
Now, plot the scoring history,

``` python
best_ntrees_rf_model.plot(metric='auc')
```
![overfitting-rf](assets/overfitting-rf.jpg)

Our model has started overfitting to the training data, and that’s why even if you use more trees, the training AUC will keep increasing, but the validation AUC will remain the same. For that reason, one way to find a good number of trees is to just make a model with a large number of trees, and from the scoring plot identify a good cut-off to find the right number of trees for your model (please keep in mind that you need to be doing cross validation for this) or just use early stopping.
From the scoring history plot we can see that the Validation AUC starts plateauing around 200 trees, so we will use that. H2O models are by default optimized to give a good performance; therefore, sometimes there is not much tuning to be done. We will see that with the GBM model as well.

To do the grid search for the RF model in Flow, start by building another default RF model the same way we did in Task 5. But this time, check the *Grid* checkbox next to `max_depth` and `ntrees` and add the list of the hyper-parameters that you want. 
For example, for **ntrees** you can use the following list: `10; 50; 70; 100; 300; 400;`
And for **max_depth** you can do `1;5;10;12;15;20;50;` 
After you click on build, your grid search for your RF will start. You can even use values closest to the one we obtained in the Jupyter Notebook to see if you can get an even better score! 

Going back to the Jupyter Notebook, let’s build the new model with max_depth of 12 and 200 trees based on our findings from the grid search. 

```python
rf = H2ORandomForestEstimator(max_depth=12, ntrees=200,
                              model_id='best_rf', seed=42)
%time rf.train(x=x, y=y, training_frame=train, validation_frame = valid)
```
Print the validation AUC 
```
tuned_rf_per = rf.model_performance(valid)
tuned_rf_per.auc()
```
And the F1 Score

```python
tuned_rf_per.F1()
```
The AUC from the validation data was **0.8521** and the F1 Score was **0.3018**

We will compare the tuned model with the default model. 
``` python
print("Default RF AUC: %.4f \nTuned RF AUC:%.4f" % (rf_default_per.auc(), tuned_rf_per.auc()))
```
Output
```
Default RF AUC: 0.8263 
Tuned RF AUC:0.8521
```
The AUC value for our RF model did increase by changing the max_depth and the number of trees. Let’s see if the F1 Score improved,
```python
print("Default RF F1 Score:", default_rf_per.F1())
print("Tuned RF F1 Score:", tuned_rf_per.F1())
```

``` 
Default RF F1 Score: [[0.18667329417807715, 0.2828354251369984]]
Tuned RF F1 Score: [[0.1403858028297814, 0.3018429354142454]]
```
The F1 score also improved. Since the F1 Score and the confusion matrix are closely related, let’s see how this improvement reflects on the confusion matrix

``` python
print ("Default RF: ", rf_default_per.confusion_matrix())
print ("Tuned RF: ",  tuned_rf_per.confusion_matrix())
```

![rf-default-vs-tuned-cf-mx](assets/rf-default-vs-tuned-cf-mx.jpg)

The AUC for our tuned model actually improved, as well as the F1 Scored. However, the misclassification error slightly increased. The new model is predicting fewer FALSE labels that are actually FALSE, and is also predicting more TRUE labels as FALSE, and for that reason the misclassification error for the FALSE predicted label slightly increased. On the other hand, the error for the TRUE predicted label decreased, and that is because the model is predicting more TRUE labels that are actually TRUE. It is good to see that the model is now predicts more TRUE labels as TRUE, because we saw that the default model, as well as the GLM were also having a hard time doing those predictions. 
Now, we will see if we can improve our GBM model.
## Task 9: Tuning the GBM model with H2O GridSearch 
We will take a similar approach to the tuning of the RF model. We could do the grid search for a list of a number of trees, but since the scoring history will show us the validation score based on the number of trees, we will obtain that number from the plot. We will be using 150 trees just so that we can see the different number of trees being used. For a GBM model, conceptually speaking, the **max_depth** and **ntrees** is the same as the RF model. However, we will see that the values are smaller than the ones used for the RF.

``` python
hyper_params = {'max_depth' : [1,3,5,6,7,8,9,10,12,13,15],
               }

gbm = H2OGradientBoostingEstimator(model_id='grid_gbm', ntrees=150,
    seed=42
    )

gbm_grid = H2OGridSearch(gbm, hyper_params,
                         grid_id = 'depth_gbm_grid',
                         search_criteria = {
                             "strategy":"Cartesian"})

%time gbm_grid.train(x=x, y=y, training_frame=train, validation_frame = valid)
```
Print the models,

``` python
sorted_gbm_depth = gbm_grid.get_grid(sort_by='auc',decreasing=True)
sorted_gbm_depth
```
Based on the grid search that we just did, the best max_depth is 5. 
We will take a look at the best model scoring history plot, and get the number of trees from there. 

![gbm-scoring-history-150](assets/gbm-scoring-history-150.jpg)

It looks like from 60 to 150 trees, the validation score keeps slightly improving. Since it will not be a great difference, let us use 80 trees, with max_depth of 5. To get the validation AUC of the model that we just decided on, we can quickly build the model.

```python
gbm = H2OGradientBoostingEstimator(max_depth=5, ntrees=80,
    seed=42, model_id='tuned_gbm'
    )
%time gbm.train(x=x, y=y, training_frame=train, validation_frame = valid)
gbm.plot(metric='auc')
```
```
gbm_per = gbm.model_performance(valid)
print(gbm_per.auc())
print(gbm_per.F1())
```
The scores from the quick grid search that we did yielded an AUC of **0.8570** and an F1 Score of **0.3023**

We will check if a random search for other parameters could improve the score.

Note: You don’t have to run the following line of code unless you want to see it for yourself. The search criteria will only allow the grid search to run for 15 minutes, if you would like to see the results of running it for longer, just increase the ‘max_runtime_secs’ to a higher value and wait for the results. 

Here is the list of parameters that we are going to try to tune,
- **sample_rate:** Specify the row sampling rate (x-axis). (Note that this method is sample without replacement.) The range is 0.0 to 1.0, and this value defaults to 1. Higher values may improve training accuracy. Test accuracy improves when either columns or rows are sampled.

- **col_sample_rate:** Specify the column sampling rate (y-axis). (Note that this method is sampling without replacement.) The range is 0.0 to 1.0. 

- **col_sample_rate_per_tree:** Specify the column sample rate per tree. This can be a value from 0.0 to 1.0 and defaults to 1. Note that it is multiplicative with col_sample_rate, so setting both parameters to 0.8, for example, results in 64% of columns being considered at any given node to split.

- **col_sample_rate_change_per_level:** This option specifies to change the column sampling rate as a function of the depth in the tree.

- **learn_rate:** Specify the learning rate. The range is 0.0 to 1.0.

- **nbins:** Specify the number of bins for the histogram to build, then split at the best point.

- **nbins_cats:** Specify the maximum number of bins for the histogram to build, then split at the best point. Higher values can lead to more overfitting.

- **min_split_improvement:** The value of this option specifies the minimum relative improvement in squared error reduction in order for a split to happen.

- **histogram_type:** Random split points or quantile-based split points can be selected as well. RoundRobin can be specified to cycle through all histogram types (one per tree). Use this option to specify the type of histogram to use for finding optimal split points

Find more parameters at the [Documentation - GBM Section](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html#gradient-boosting-machine-gbm) and also the [Python Module](https://h2o-release.s3.amazonaws.com/h2o/rel-wheeler/4/docs-website/h2o-py/docs/modeling.html#h2ogradientboostingestimator)

``` python
gbm = H2OGradientBoostingEstimator(
    max_depth=5,
    ntrees=80,
    seed=42,
    model_id='gbm_final_grid'
    )

hyper_params_tune = {
                'sample_rate': [x/100. for x in range(20,101)],
                'col_sample_rate' : [x/100. for x in range(20,101)],
                'col_sample_rate_per_tree': [x/100. for x in range(20,101)],
                'col_sample_rate_change_per_level': [x/100. for x in range(90,111)],
                'learn_rate' : [.5, .25, 0.1, 0.07, 0.05, 0.01, 0.001],
                'nbins': [2**x for x in range(4,11)],
                'nbins_cats': [2**x for x in range(4,13)],
                'min_split_improvement': [0,1e-8,1e-6,1e-4],
                'histogram_type': ["UniformAdaptive","QuantilesGlobal","RoundRobin"]}

search_criteria_tune = {'strategy': "RandomDiscrete",
                   'max_runtime_secs': 900,  
                   'max_models': 100,  ## build no more than 100 models
                   'seed' : 42 }

random_grid = H2OGridSearch(model=gbm, hyper_params=hyper_params_tune,
                         grid_id = 'random_grid',
                         search_criteria =search_criteria_tune)

%time random_grid.train(x=x, y=y, training_frame=train, validation_frame = valid)
``` 

Let’s see if the random search yielded any improvement:

``` python
best_gbm_model = random_grid.models[0] 
tuned_gbm_per = best_gbm_model.model_performance(valid)
print(tuned_gbm_per.auc())
print(tuned_gbm_per.F1())
```

The random search yielded an AUC of **0.8581** and an F1 Score of **0.3140.** The random search did yield to an improvement on the AUC and the F1 score, running the grid search for longer time could further improve the metrics scored. For now, we can leave it at that, or you can try it on your own to see if you get better results!

If you go to your Flow instance, you can check your best model. See the results for yourself and compare it to the results that we obtained from the default model. You can also look at the Variable importance plot. The variable importance plot seems very similar to the one we obtained for the default GBM model, except for the 5th predictor which has changed. You can also look at the confusion matrix on both training and validation data. 

We will do the final test performance next.
## Task 10: Test Set Performance
We are going to obtain the test performance of each of the best models. If you named your models the same as in this tutorial, then you should be able to just run the following code. Notice that we are just taking the best models, and checking the model performance with the test set. 

``` python
glm_test_per = best_glm_model.model_performance(test)
rf_test_per = rf.model_performance(test)
gbm_test_per = best_gbm_model.model_performance(test)
```
You can now print any performance metric that you would like. Right now we will just focus on the AUC, F1 Score, and the misclassification error from the confusion matrix. 

Print the test AUC of each model. 

``` python
print("GLM Test AUC: %.4f \nRF Test AUC: %.4f \nGBM Test AUC: %.4f " % 
      (glm_test_per.auc(), rf_test_per.auc(), gbm_test_per.auc()))

```
 Output:
``` 
GLM Test AUC: 0.8524 
RF Test AUC: 0.8565 
GBM Test AUC: 0.8619
```
We were able to improve the AUC of all three models with the quick grid search that we did for all three models. We saw the greatest improvement with the RF model, as the default parameters were a little off from what we found to be optimal. All three AUC test scores are slightly higher than the validation scores but close enough to trust the validation score to tune all our models. And as it could be expected, the GBM had the best AUC, followed by the RF and lastly, the GLM. 

Now print the F1 Score for each model,

```python
print ("GLM Test F1 Score: ", glm_test_per.F1())
print ("RF Test F1 Score: ",  rf_test_per.F1())
print ("GBM Test F1 Score: ",  gbm_test_per.F1())
```
Output:
```
GLM Test F1 Score:  [[0.13264950036559808, 0.28271196199798476]]
RF Test F1 Score:  [[0.1308861956227085, 0.2943511450381679]]
GBM Test F1 Score:  [[0.14395615838808512, 0.3034588777863182]]
```
The F1 Score for the RF and GBM slightly increased compared to the default value; however, the GLM F1 Score actually slightly decreased compared to both the default and the validation results. Even though the AUC for the GLM improved, the F1 did not, and we will see shortly how that is reflected in the misclassification error. On the other hand, by tuning some parameters, we were able to get better AUC and better F1 Score for both RF and GBM models. 

Lastly, we will take a look at the confusion matrix for each model

``` python
print ("GLM Confusion Matrix: ", glm_test_per.confusion_matrix())
print ("RF Confusion Matrix: ",  rf_test_per.confusion_matrix())
print ("GBM Confusion Matrix ",  gbm_test_per.confusion_matrix())
```

![confusion-matrix-test](assets/confusion-matrix-test.jpg)

Again, all three scores are very close to each other, and the best one being the GBM, second the RF, and lastly our GLM. For the Misclassification error, we see the opposite pattern to the F1 Score, the misclassification error for both RF and GBM increased, and it slightly decreased for the GLM. However, it is important to note that for both RF and GBM the error for the TRUE predicted label decreased, and for the GLM increased. This, along with a relatively low F1 Score is due to the highly imbalanced dataset. 

For this dataset, we obtained a good AUC for all three models. We obtained an okay F1 Score, given that our dataset is highly imbalanced, and we also obtained a good overall misclassification error, although due to the given imbalanced data, the error for the TRUE label was not so low. Overall, The best model trained on our dataset was the GBM, followed by the RF, and lastly the GLM.
## Task 11: Challenge & Shutting down your Cluster
After building three models, you are now familiar with the syntax of H2O-3 models. Now, try to build a Naive Bayes Classifier! We will help you by showing you how to import the model. The rest is up to you. Try it and see what’s the best training, validation and test AUC that you can get with the Naive Bayes Classifier and compare it to the models that we built in this tutorial.


``` python
from h2o.estimators import H2ONaiveBayesEstimator
```

Once you are done with the tutorial, remember to shut down the cluster.

``` python
h2o.cluster().shutdown()
```

## Task 12: Next Steps

Introduction to Machine Learning with H2O - Part 2 coming soon


