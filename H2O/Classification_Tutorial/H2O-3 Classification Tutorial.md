Classification Tutorial with H2O-3

Overview:
Dataset: The H2O Freddie_Mac subset 
Models that will be used: GLM, Random Forest, and GBM
Prediction: will a loan be delinquent or not

Outline
Objective
Pre-requisites 
Preview
Task 1: Import H2O, libraries and models. Initialize cluster (introduce Flow) and load the data.
Task 2: Quick look at the data, split the data, and choose predictors and response
Task 3: Build a GLM with default settings, and inspect results
Task 4: Build a Random Forest with default settings, and inspect results
Task 5: Build a GBM with default settings, and inspect results
Task 6: Grid Search for GLM
Task 7: Grid Search for RF
Task 8: Grid Search for GBM
Task 9: Use the best models, along with test set and check results
Task 10: Challenge and Shutting down the Cluster
Next Steps: Regression Tutorial Coming Soon








## Objective  
We will be using a subset of the Freddie Mac Single-Family dataset to try to predict if a loan will be delinquent or not using H2O’s GLM, Random Forest, and GBM models. We will go over how to use these models for classification problems, and we will do a grid search with H2O’s grid search, in order to tune the hyperparameters of each model.

## Pre-requisites 
Some basic knowledge of machine learning. Familiarity with Python. Make sure you have Jupyter Notebook installed on your local machine, and that you already installed H2O-3.

If you do not have H2O-3, you can follow the installation guide on the [H2O Documentation page](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html)

We recommend creating an Anaconda Cloud environment, as shown in the installation guide, [Install on Anaconda Could.](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html#install-on-anaconda-cloud) This would guarantee that you will have everything that you need to do this tutorial. 

OR

Follow along on Aquarium -  cloud instance …
## Preview
The data set we’re using comes from Freddie Mac - it has mortgage history for 20 years - for each loan, the loanholder’s information, payment status etc. This dataset contains information about "loan-level credit performance data on a portion of fully amortizing fixed-rate mortgages that Freddie Mac bought between 1999 to 2017. Features include demographic factors, monthly loan performance, credit performance including property disposition, voluntary prepayments, MI Recoveries, non-MI recoveries, expenses, current deferred UPB and due date of last paid installment."[1] 
We’re going to use machine learning with H2O to predict whether someone will default or not. To do this we are going to build three classification models, a Linear Model, Random Forest, and a Gradient Boosting Machine, to predict if a loan will be delinquent or not. Complete this tutorial to see we achieved those results.

[1] Our dataset is a subset of the Freddie Mac Single-Family Loan-Level Dataset. It contains about 500,000 rows and is about 80 MB.
## Task 1 - Import H2O, libraries, and models. Initialize H2O and load the dataset.
We will start by importing H2O, the estimators for the algorithms that we will use, and also the function to perform Grid Search  on those algorithms. 

``` python
#Import H2O and other libraries that will be used in this tutorial 
import h2o
import matplotlib as plt

#Import the Estimators
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2OSupportVectorMachineEstimator

#Import h2o grid search 
import h2o.grid 
from h2o.grid.grid_search import H2OGridSearch
```
We now have to initialize an h2o cluster or instance, in this  case. You can specify how much memory you want your cluster to have. We will assign 4Gb of memory, although we won’t be using all of it all. Typically we recommend a H2O Cluster with at least 3-4 times the amount of memory as the dataset size. 

``` python
h2o.init(max_mem_size="4G")
```

![cluster-info](assets/cluster-info.jpg)


Basically we’ve a H2O cluster with just 1 node. After initializing the H2O cluster, you will see the information shown above. Clicking on the link will take you to your **Flow instance** where you can see your models, data frames, plots and much more. Click on the link, and it will take you to a window similar to the one below. Keep it open in a separate tab, as we will come back to it later on.

![flow-welcome-page](assets/flow-welcome-page.jpg)



Next, we will import the dataset. You can download H2O's subset of the Freddie Mac Single-Family Loan-Level dataset to your local drive and save it at as csv file. Loan_level_500k.csv.
Make sure that the dataset is in the same directory as your Jupyter Notebook. For example, if your Jupyter file is in your **Documents** save the csv file there. Or, you can just specify the path of where the file is located; in our case the file is in S3. That’s why we’ll just do the following: 

``` python
#Import the dataset 
loan_level = h2o.import_file("https://s3.amazonaws.com/data.h2o.ai/DAI-Tutorials/loan_level_500k.csv")
Parse progress: |█████████████████████████████████████████████████████████| 100%
```

Now that we have our dataset, we will look at it, and do some preparation for our models.
## Task 2 - Quick look at the data, data split, and predictor and response selection.

Make sure the dataset was properly imported by using the `.head()` command 

```
loan_level.head()
```
![dataset-head](assets/dataset-head.jpg)


You should be able to see a table like the one above. You can keep scrolling to the right to see all the features in our dataset. 
We can also take a quick look at the description of our dataset with the `.describe()` command as shown below

``` python
loan_level.describe()
```

![data-describe](assets/data-describe.jpg)

The total number of rows in our dataset is 500,137 and the total number of features or columns is  27. With the description displayed, you can see the type of your column, the minimum and maximum values in each column. Also, the number of missing values in each column, among other characteristics. 

You can also do the same thing with H2O Flow, by clicking ‘import’ and then viewing the actual table once it’s imported. Go to your Flow instance and add a new cell

![flow-add-cell](assets/flow-add-cell.jpg)

Copy and paste the following line of code in the new cell and run it. Then, click on **Parse these files** 

```
importFiles ["https://s3.amazonaws.com/data.h2o.ai/DAI-Tutorials/loan_level_500k.csv"]
```

![flow-parse-file](assets/flow-parse-file.jpg)

After clicking on **Parse these files**, you will see a parse set-up similar to the image below

![flow-parse-set-up](assets/flow-parse-set-up.jpg)


H2O will try to parse the file and assign  appropriate  column types. But you change columns types if they’re not imported correctly. After you have inspected the parse set-up, click on parse. 

Once finished, you will see the following message, confirming that the parsing was completed. 















![flow-parse-finished](assets/flow-parse-finished.jpg)


**The main goal of this tutorial is to show the usage of some models for classification problems, as well as to tune some of the hyperparameters of the models. For that reason, we will be skipping any data visualization, and manipulation, as well as the feature engineering. The aforementioned stages in machine learning are very important, and should always be done; however, they will be covered in later tutorials.**

Since we have a large enough dataset, we will split our dataset into three sets and we will call them **train, valid,** and **test.** We will use the valid set for validation purposes and to tune all our models. We will not use the test set until the end of the tutorial to check the final scores of our models. 

Return to your Jupyter Notebook to split our dataset into three sets. We will use the `.split_frame ()` command. Note that we can do this in one line of code. Inside the split function, we declare the ratio of the data that we want in our first set, in this case, **train** set. We will assign 70% to the training set, and 15% for the validation, as well as for the test set. The random seed is set to 42 just for reproducibility purposes. You can choose any random seed that you want, but if you want to see the consistent results, you will have to use the same random seed anytime you re-run your script. 

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



Next, we need to choose our predictors, or x variable, and our response or y variable. For the H2O-3 models, we do not use the actual data frame, but strings containing the name of the columns in our dataset.



For our y variable, we will choose `DELINQUENT` because we want to predict if a loan will be delinquent or not. For the x variable, we will choose all but 4 features. One is the feature that we will predict, and then `PREPAID` and `PREPAYMENT_PENALTY_MORTGAGE_FLAG` because they are clear indicators if a loan is delinquent or not and we will not have the information at the time deciding whether to give a loan out or not. In machine learning terms, introducing these type of features is called leakage.. And lastly, `PRODUCT_TYPE` because that’s a constant value for every row, meaning all samples have the same value; therefore, this feature will not help us at all.

There are several ways to choose your predictors, but for this tutorial, we will use a **for** loop that will ignore the four features mentioned above. 

``` python
y = "DELINQUENT"

ignore = ["DELINQUENT", "PREPAID", "PREPAYMENT_PENALTY_MORTGAGE_FLAG", "PRODUCT_TYPE"] 

x = [i for i in train.names if i not in ignore]
```

If you want to see the list of the features that are in your x variable, just print x.

``` python
print(x)
```
```
['CREDIT_SCORE', 'FIRST_PAYMENT_DATE', 'FIRST_TIME_HOMEBUYER_FLAG', 'MATURITY_DATE', 'METROPOLITAN_STATISTICAL_AREA', 'MORTGAGE_INSURANCE_PERCENTAGE', 'NUMBER_OF_UNITS', 'OCCUPANCY_STATUS', 'ORIGINAL_COMBINED_LOAN_TO_VALUE', 'ORIGINAL_DEBT_TO_INCOME_RATIO', 'ORIGINAL_UPB', 'ORIGINAL_LOAN_TO_VALUE', 'ORIGINAL_INTEREST_RATE', 'CHANNEL', 'PROPERTY_STATE', 'PROPERTY_TYPE', 'POSTAL_CODE', 'LOAN_SEQUENCE_NUMBER', 'LOAN_PURPOSE', 'ORIGINAL_LOAN_TERM', 'NUMBER_OF_BORROWERS', 'SELLER_NAME', 'SERVICER_NAME']
``` 
## Task 3 - Build a GLM with default settings and inspect the results

Now that we have our train, valid, and test set, as well as our x and y variables, we can start building models! We will start with an H2O Generalized Linear Model (GLM). A GLM fits a generalized linear model, specified by a response variable, a set of predictors, and a description of the error distribution. Since we have a binomial classification problem, we have to specify the family, in this case, it will be “binomial.” 

Since we already imported the H2O GLM estimatormodel, we will just instantiate our model. For simplicity, the name of our model will be `GLM`. To build a GLM, you just need to define the family and you are ready to go. You can instantiate your GLM as shown below. 

``` python
GLM = H2OGeneralizedLinearEstimator(family = "binomial")
```
Now we will train our glm model. To do so, we just use the `.train()` function. In the train function, we need to specify the predictors (x), the response (y), your training set (train) and a validation_frame, if you have one. In our case, we have our valid set, which we will use. 

``` python
%time GLM.train(x, y, train, validation_frame = valid)
```
The `%time` in front of our train command is only used to display the time it takes to train the model. 


You can do the same thing from Flow with the ‘Build model’ dialog. 

You have now built and trained a GLM! If you type the name of your model in a new cell and run it, H2O will give you a complete summary of your model. You will see your model’s metrics on the training and validation set. 






Add Variable importance chart here and highlight which features are most significant 


From the summary results, we can see that the GLM has a good performance. 
Since this is a classification problem, we will focus on two of the metric scores, which are the accuracy and the Area Under the Curve (AUC). We would also take a quick look at the misclassification error and logloss. 

To learn more about AUC, ROC etc. go here: (add that section here, maybe)

From the report above, we saw that the training AUC was 0.8503712628375173, while the validation AUC was 0.8451146381798539

In the report, you can take a look at the `max accuracy` for the accuracy of this model. The training accuracy was 0.9638505 while the validation accuracy was 0.9645330

We will take a look at the first ten predictions of our model with the following command:

``` python
GLM.predict(valid)
``` 

The model used for this classification problem is a Logistic Regression model. The predictions are based on the threshold for the probability. Based on the report, the threshold is about 0.123. So, any time the probability for TRUE is greater than the threshold, the prediction will be TRUE. As is in the case of the sixth prediction. 

Go back to the Flow tab that you opened in task 1, or just click on the link shown in your cluster information. Click on **getModels**, you should be able to see the model you just built. 

![flow-get-model](assets/flow-get-model.jpg)

Click on it, and if you expand the `Model Parameters` you will see a screen similar to the one below,

![flow-glm-params]

If you keep scrolling down, you will see the `Scoring History` plot. 

![flow-glm-scoring-history]
This plot tells us that after the third iteration, the score does not improve, but just remains constant. If we want to define a number of iterations, we can choose 3, based on this plot. 

If you keep scrolling down, you will see the ROC curves, both for training and validation. You will also see the AUC score for both training and validation. You can change the threshold and criterion for your ROC curve, and see how your model performs with different thresholds and at different points. 

![flow-rocs-glm]
Below, you will see all the options you can explore in Flow about your GLM

![flow-complete-list]

After the short Flow tour, we will build the next model. 
## Task 4 - Build a Random Forest with default settings and inspect the initial results
We will build a default Distributed Random Forest (DRF) model and see how it performs on our validation set. DRF generates a forest of classification or regression trees, rather than a single classification or regression tree. Each of these trees is a weak learner built on a subset of rows and columns. More trees will reduce the variance. Both classification and regression take the average prediction over all of their trees to make a final prediction, whether predicting for a class or numeric value. 

To build and train our Random Forest (RF) model, simply run the following two lines of code:
``` python
RF = H2ORandomForestEstimator (seed=42, model_id='random_forest')
%time RF.train(x, y, train, validation_frame=valid)
```
Note that we defined the random seed and the model id. You do not need to do this, the model can be built without defining these parameters. The reason for choosing the random seed is for reproducibility purposes, and the model id is to easily recognize the model in Flow. 

Again, print the summary of your model as we did with the GLM model. You will see the summary of the model with the default settings, and the metrics score on the training and validation set. 




If you would like, you can go to Flow and inspect the outputs of your RF model. However, we can also generate the plots in our Jupyter Notebook

``` python
RF.plot(metric='auc')
```
You will see a plot similar to the one below

![rf-scoring-history](assets/rf-scoring-history.jpg)

In this case, we see that the RF model is far from overfitting because the training error is still lower than the validation error and that means that we can probably do some tuning to improve our model. 

We can also generate the variable importance plot,
```python
RF.varimp_plot(20)
```

![rf-var-imp](assets/rf-var-imp.jpg)


It is interesting to see that for our RF model, `PROPERTY_STATE` Is the most important variable, implying that the prediction of whether a loan could be delinquent or not depends on the state where someone is trying to buy that property. The second most important is a more intuitive one, which is the CREDIT_SCORE, as one could expect someone with really good credit to fully pay their loans. 

Print the model summary to some of the parameters of our default model.

``` python
RF.summary().as_data_frame()
```



We will use this information when we start tuning our models. If you want to check the options of what you can print for your model, just type the name of your model along with a dot (“.”) and press tab. You should see a drop-down menu like the one shown in the image below. 


Keep in mind that for some of those you will need to open and close parentheses at the end in order to display what you want. Let’s say we wanted to print the training accuracy of our model, you could select accuracy, but you need to add parentheses in order to get just the accuracy, otherwise, you will get the entire report again.


The first parameter shown in the list above is the threshold, and the second value is the accuracy. 

Let’s take a look at the first ten predictions in our validation set, and compare it to our first model.

``` python
RF.predict(valid)
```

Both models made the same predictions in the first ten predictions. For e.g. Even the TRUE prediction, for the sixth row; there is a different probability, but the prediction is the same. 
## Task 5: Build a GBM with default settings

Gradient Boosting Machine (for Regression and Classification) is a forward learning ensemble method. H2O’s GBM sequentially builds classification trees on all the features of the dataset in a fully distributed way - each tree is built in parallel. H2O’s GBM fits consecutive trees where each solves for the net loss of the prior trees. 
Sometimes GBMs tend to be the best possible models because they are robust and directly optimize the cost function. On the other hand, they tend to overfit, so you need to find the proper stopping point; they are sensitive to noise, and they have several hyper-parameters.

Defining a GBM model is as simple as the other models we have been working with. 
``` python
GBM = H2OGradientBoostingEstimator()
%time GBM.train(x, y, train, validation_frame = valid)
``` 
Print the model summary






Go to Flow, and take a look at the scoring history. 



Scroll down to the variable importance plot, and take a look at it. Notice how the most important variable is `CREDIT_SCORE` for the GBM. If you recall, for Random Forest, `CREDIT_SCORE` was the second most important variable. And the most important variable for Random Forest is the third most important for the GBM. 

The default GBM model had a slightly better performance than the default random forest. 
We will do the prediction with the GBM model as well, as we did with the other two models. 

``` python
GBM.predict(valid)
``` 


All three models made the same 10 predictions and this gives us an indication of why all three scores are close to each other. Although the sixth prediction is TRUE for all three models, the probability is not exactly the same, but since the thresholds for all three models were low, the predictions were still TRUE. 
Next, we will tune our models and see if we can achieve better performance. 
## Task 6: Tuning the GLM with H2O GridSearch. 
H2O supports two types of grid search – traditional (or “cartesian”) grid search and random grid search. In a cartesian grid search, you specify a set of values for each hyperparameter that you want to search over, and H2O will train a model for every combination of the hyperparameter values. This means that if you have three hyperparameters and you specify 5, 10 and 2 values for each, your grid will contain a total of 5*10*2 = 100 models.

In random grid search, you specify the hyperparameter space in the exact same way, except H2O will sample uniformly from the set of all possible hyperparameter value combinations. In the random grid search, you also specify a stopping criterion, which controls when the random grid search is completed. You can tell the random grid search to stop by specifying a maximum number of models or the maximum number of seconds allowed for the search. You can also specify a performance-metric-based stopping criterion, which will stop the random grid search when the performance stops improving by a specified amount.
Once the grid search is complete, you can query the grid object and sort the models by a particular performance metric (for example, “AUC”). All models are stored in the H2O cluster and are accessible by model id. 

To save some time, we will do a random grid search for our GLM model instead of the cartesian search. The H2OGridSearch has **4 parameters,** and in order to use it, you need **at least three** of them. The first parameter for the grid search is the **model** that you want to tune. Next are your **hyperparameters,** which needs to be a string of parameters, and a list of values to be explored by grid search. The third one is optional, which is the **grid id,** and if you do not specify one, an id will automatically be generated. Lastly, the **search criteria,** where you can specify if you want to do a cartesian or random search.  

We will explore two ways of defining your grid search, you can use the way you prefer. One way is to define all at once in the grid search (as we will do it for the GLM). The second way is to define every parameter separately. For example, define your model, your hyper-parameters, and your search criteria, and just add that to your grid search once you are ready.

For our GLM, we can only tune **alpha** and **lambda.** The other parameters that you could change, such as *solver,* *max_active_predictors,* and *nlambdas* to mention a few, are not supported by H2OGridSearch. 

Alpha is the distribution of regularization between the L1 (Lasso) and L2 (Ridge) penalties. A value of 1 for alpha represents Lasso regression, a value of 0 produces Ridge regression, and anything in between specifies the amount of mixing between the two. Lambda, on the other hand, is the regularization strength. For alpha, we can explore the range from 0 to 1 in steps of 0.01. For lambda, you could start just doing your own random searches, but that would be a waste of time. Instead, we can base our guess for lambda on the original value of lambda, which was 6.626e-5. We can choose our starting point to be 0.1e-5, and go from there. The example of this is shown below. 

``` python
glm_grid = h2o.grid.H2OGridSearch (
    H2OGeneralizedLinearEstimator( 
        family = "binomial",
        lambda_search = True),
    hyper_params = {
        "alpha": [x*0.01 for x in range(0, 100)],
        "lambda": [x*.1e-5 for x in range(0, 1000)],
        },
    grid_id = "glm_grid",
    search_criteria = {
        "strategy":"RandomDiscrete",
        "max_models":50,
        "max_runtime_secs":300
        }
    )
%time glm_grid.train(x, y, train, validation_frame = valid)
```
 You can easily see all four parameters for our grid search in the code sample above. We defined our glm model the same way we did before. Notice that we have used a for loop for the ranges of both alpha and lambda. Because the number of possible models is really big, in our search criteria, we specify that we want a maximum number of 50 models, or that the grid search runs for only 300 seconds.
 
Now, we will print the models in descending order, sorted by the AUC. By default, the grid will return the best models based on the `logloss`. Therefore, in order to get the best model based on the AUC, we will specify that we want to sort the models by AUC.

``` python
sorted_glm_grid = glm_grid.get_grid(sort_by='auc',decreasing=True)
sorted_glm_grid
```
With the code sample above, you will get the models that were created, with their respective alpha, lambda, model id, and AUC. If you want more details about each model, you can go to Flow, click on “getModels” and you can easily find all the models by their model id. 

Let's explore the best model obtained from our grid search. Save the best model, and print the model summary with the following code:

``` python
best_glm_model = glm_grid.models[0] 
best_glm_model.summary().as_data_frame()
``` 
The second line of code will print the parameters used for the best model found by the grid search. We will do a quick comparison between the performance of the default glm model and the best model from the grid search. 
First, evaluate the model performance on the validation set.

```python
glm2_perf = best_glm_model.model_performance(valid)
``` 
Now, print the AUC for the default, and the tuned model.

``` python
print("Default GLM AUC: %.5f \nTuned GLM AUC:%.5f" % (glm_perf.auc(), glm2_perf.auc()))
```
Output:
``` 
Default GLM AUC: 0.84511 
Tuned GLM AUC:0.84611
``` 
The AUC did not really improve. Statistically, it would not be considered an improvement, but it slightly changed. We did not expect the GLM model to perform great, or to have a great improvement with the grid search, as it is just a linear model, and in order to perform good, we would need a linear distribution of our response. 

We can print the accuracy to see if it changed at all.
``` python
print ("Default GLM Accuracy:", glm_perf.accuracy())
print ("Tuned GLM Accuracy", glm2_perf.accuracy())
```
Output:
``` Text
Default GLM Accuracy: [[0.9658777656085674, 0.9645329527417268]]
Tuned GLM Accuracy [[0.9741209340467776, 0.9645329527417268]]
```
The max accuracy did not change, we obtained the same value, but notice that the threshold slightly increased. We will see if the confusion matrix also changed. So, let's take a look at it. 

``` python
print ("Default GLM: ", glm_perf.confusion_matrix())
print ("Tuned GLM: ",  glm2_perf.confusion_matrix())
``` 



Notice how the overall error decreased, as well as the error for the FALSE class that was correctly classified. But the error for the TRUE class went up, meaning the model is classifying more samples that are TRUE as FALSE. 

We will do the test evaluation after we tune our other two models.



## Task 7: Tuning the RF model with H2OGridSearch 

We will do the grid search a bit differently this time. We are going to define each parameter of the grid search separately, and then add it to the grid search.

We will first find the appropriate depth for the random forest. We will do this to save some computational time when we tune the other parameters. As we mentioned before, we will use a slightly different approach for the grid search. We are going to instantiate each parameter for the grid search, and then pass each one into it.

``` python
rf_depth = {'max_depth' : [1,3,5,6,7,8,9,10,12,13,15,20]} #hyperparameter

#Model
RF2 = H2ORandomForestEstimator(  
    seed=42,
    stopping_rounds=5, 
    stopping_tolerance=1e-4, 
    stopping_metric="auc"
    )

grid_id = 'depth_grid'

search_criteria = {'strategy': "Cartesian"}  #Search Criteria

#Grid Search
rf_grid = H2OGridSearch(RF2, rf_depth, grid_id, search_criteria)

%time rf_grid.train(x, y, train, validation_frame = valid)
```
After it is done training, print the models sorted by AUC.

``` python
sorted_rf_depth = rf_grid.get_grid(sort_by='auc',decreasing=True)
sorted_rf_depth
```
Now that we have the proper depth for our random forest, we will try to tune our other parameter

The other most important parameter that we can tune is the number of trees (`ntrees`). When tuning the number of trees, you need to be careful because when you have too many trees, your model would tend to overfit. Let's take a look at it.


We will use the grid search to build models with 10, 50, 70, 100, 300, 500 and 1000 trees. 

``` python
hyper_parameters2 = {#'max_depth':[1,3,5,6,7,8,9,10,12,13,15,20,25,35],
                    'ntrees' : [10, 50, 70, 100, 300, 500, 1000]}

RF3 = H2ORandomForestEstimator(max_depth=12,
    seed=42,
    stopping_rounds=5, 
    stopping_tolerance=1e-4, 
    stopping_metric="auc"
    )
grid_id = 'ntrees_grid'
search_criteria = {'strategy': "Cartesian"}

rf_grid = H2OGridSearch(model=RF3, 
                        hyper_params=hyper_parameters2, grid_id=grid_id, search_criteria=search_criteria)

%time rf_grid.train(x, y, train, validation_frame = valid)
```

``` python
sorted_rf_ntrees = rf_grid.get_grid(sort_by='auc',decreasing=True)
sorted_rf_ntrees
```
``` text
   ntrees            model_ids                 auc
0     1000  ntrees_grid_model_7  0.8534535906875113
1      500  ntrees_grid_model_6  0.8530095611964112
2      300  ntrees_grid_model_5  0.8529632779921271
3      100  ntrees_grid_model_4  0.8513762260847957
4       70  ntrees_grid_model_3  0.8503948619108656
5       50  ntrees_grid_model_2  0.8495585203513482
6       10  ntrees_grid_model_1  0.8374501192598929
```
By looking at the training scores, we see that the best model would be the one with 1000 trees! Let’s build a model with max_depth of 12, and with 1000 trees. 

``` python
RF3 = H2ORandomForestEstimator(max_depth=12, ntrees=1000,
    seed=42)
%time RF3.train(x, y, train, validation_frame = valid)
```
Now, plot the scoring history,

``` python
RF3.plot(metric='auc')
```


Our model has started overfitting to the training set, and that’s why even if you use more trees, the training AUC will keep increasing, but the validation AUC will remain the same. For that reason, one way to find a good number of trees is to just make a model with a large number of trees, and from the scoring plot identify a good cut-off to find the rightobtain a good number of trees for your model (please keep in mind that you need to be using a validation set to do this).

Here  we can see that the Validation AUC starts plateuing around 200 trees, Looks like 200 trees are a good number of trees, so we will use that. H2O models are by default optimized to give a good performance; therefore, sometimes there is not much tuning to be done, we will see that with the GBM model.

We will compare the tuned model with the default model. 
``` python
print("Default RF AUC: %.5f \nTuned RF AUC:%.5f" % (rf_default_per.auc(), tuned_rf_per.auc()))
```
You can also print the confusion matrix and see if you had any major improvement 

``` python
print ("Default RF: ", rf_default_per.confusion_matrix())
print ("Tuned RF: ",  tuned_rf_per.confusion_matrix())
```
The AUC for our tuned model actually improved; however, the misclassification error didn't really change much. The model is predicting fewer FALSE labels that are actually TRUE, and for that reason, we see a slight improvement in the misclassification error for the TRUE label. Now, we will see if we can improve our GBM model.
## Task 8: Tuning the GBM model with H2OGridSearch 
Random Forest and GBMs are quite similar, so we will take the same approach. We will take a similar approach to the tuning of the previous model. 

``` python
hyper_params = {'max_depth' : [1,3,5,6,7,8,9,10,12,13,15]}

Grid_GBM = H2OGradientBoostingEstimator(
    seed=42,
    
    )

grid = H2OGridSearch(Grid_GBM, hyper_params,
                         grid_id = 'depth_grid',
                         search_criteria = {'strategy': "Cartesian"})

%time grid.train(x, y, train, validation_frame = valid)
```
Print the models,

``` python
sorted_gbm_depth = grid.get_grid(sort_by='auc',decreasing=True)
sorted_gbm_depth
```
Instead of running a grid search to find the number of trees to use, we will build a GBM model with 500 trees, and max_depth of 6, as that gave us the best model and observe the scoring history
```python
GBM2 = H2OGradientBoostingEstimator(max_depth=6, ntrees=500,
    seed=42
    )
%time GBM2.train(x, y, train, validation_frame = valid)
GBM2.plot(metric='auc')
```

As we mentioned it earlier, GBMs tend to overfit! We will choose 80 trees and max depth of 6.

``` python
GBM3 = H2OGradientBoostingEstimator(max_depth=6, ntrees=80,
    seed=42
    )
%time GBM3.train(x, y, train, validation_frame = valid)
```  
```python
gbm_p = GBM3.model_performance(valid)
gbm_p.auc()
```  
A quick check to the model we just built. So far, this is the best model among the three. We will check if a random search could improve it a little bit more. 

Note: You don’t have to run this line of code unless you want to see it for yourself. The search criteria only allow the grid search to run for 5 minutes, if you would like to see the results of running it for longer, just increase the ‘max_runtime_secs’ to a higher value and wait for the results. 

``` python
GBM_Final_Grid = H2OGradientBoostingEstimator(
    max_depth=6,
    ntrees=80,
    seed=42
    )

hyper_params_tune = {
                'sample_rate': [x/100. for x in range(20,101)],
                'col_sample_rate' : [x/100. for x in range(20,101)],
                'col_sample_rate_per_tree': [x/100. for x in range(20,101)],
                'col_sample_rate_change_per_level': [x/100. for x in range(90,111)],
                'learn_rate' : [0.01, 0.09, 0.07, 0.05, 0.001],
                'nbins': [2**x for x in range(4,11)],
                'nbins_cats': [2**x for x in range(4,13)],
                'min_split_improvement': [0,1e-8,1e-6,1e-4],
                'histogram_type': ["UniformAdaptive","QuantilesGlobal","RoundRobin"]}

search_criteria_tune = {'strategy': "RandomDiscrete",
                   'max_runtime_secs': 300,  ## limit the runtime to 5 minutes
                   'max_models': 100,  ## build no more than 100 models
                   'seed' : 42 }
random_grid = H2OGridSearch(GBM_Final_Grid, hyper_params_tune,
                         grid_id = 'random_grid',
                         search_criteria =search_criteria_tune)

%time random_grid.train(x, y, train, validation_frame = valid)
``` 
You can find a list of all parameters in the Documentation section of GBM, and also in the GBM section of the python module

[Documentation](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html)

[Python module](https://h2o-release.s3.amazonaws.com/h2o/rel-wheeler/4/docs-website/h2o-py/docs/modeling.html#h2ogradientboostingestimator)


Let’s see if the random search yielded any improvement

``` python
best_gbm_model = random_grid.models[0] 
random_per = best_gbm_model.model_performance(valid)
random_per.auc()
```
``` 
0.8581638015875244
```
The AUC slightly increased after the running the grid search for 5 minutes. Try running it for longer on your own, and see if you get an even better model.


Let's look at this model in Flow. Look for your model based on the model id of the best model. Once you find it, check out the parameter values; you can see all the parameters that your model is using, along with a short description of it. You will also see the scoring history, with the logloss as the metric, because that is the default metric for classification problems. 
You will see the training and validation ROC curve, along with the AUC as shown below

You can also see the Variable importance plot. The variable importance plot seems very similar to the one we obtained for the default GBM model, except for the 5th predictor which has changed. You can also look at the confusion matrix on both training and validation data. 
We will do the final test performance now.


## Task 9: Test set Performance
We are going to get the test performance of each of the best models. If you named your models the same as in this tutorial, then you should be able to just run the following code. Notice that we are just taking the best models, and comparing doing the model performance with the test set. 

``` python
glm_test_per = best_glm_model.model_performance(test)
rf_test_per = RF4.model_performance(test)
gbm_test_per = best_gbm_model.model_performance(test)
```
You can now print any performance metric that you would like. In this tutorial we would just focus on the AUC, accuracy, and the misclassification error from the confusion matrix. 

Print the test AUC of each model. 

``` python
print("GLM Test AUC: %.5f \nRF Test AUC: %.5f \nGBM Test AUC: %.5f " % 
      (glm_test_per.auc(), rf_test_per.auc(), gbm_test_per.auc()))
```
 Output:
``` 
GLM Test AUC: 0.85267 
RF Test AUC: 0.85670 
GBM Test AUC: 0.86147
```
All three AUC test scores are higher than the validation scores. And as it could be expected, the GBM had the best AUC, followed by the RF and lastly, the GLM. 

We can also print the the max accuracy for each model,

``` python
print ("GLM Test Accuracy: ", glm_test_per.accuracy())
print ("RF Test Accuracy: ",  rf_test_per.accuracy())
print ("GBM Test Accuracy: ",  gbm_test_per.accuracy())
```
Output:

```
GLM Test Accuracy:  [[0.9311653657434364, 0.964311463590483]]
RF Test Accuracy:  [[0.4017307518782715, 0.9649389836844775]]
GBM Test Accuracy:  [[0.5531093319839284, 0.9649656866672007]]
```
All three maximum accuracies are very close to each other. But their thresholds are all different. The threshold for GLM is high compared to the threshold of the RF and GBM models. 

We can also look at the confusion matrix for each model

``` python
print ("GLM Confusion Matrix: ", glm_test_per.confusion_matrix())
print ("RF Confusion Matrix: ",  rf_test_per.confusion_matrix())
print ("GBM Confusion Matrix ",  gbm_test_per.confusion_matrix())
```



## Task 10: Challenge & Shutting down your Cluster
After building three models, you are now familiar with the syntax of H2O-3 models. Now, try to build a Naive Bayes Classifier! We will help you by showing you how to import the model. The rest is up to you. Try and see what’s the best training, validation and test AUC that you can get with the Naive Bayes Classifier. 


``` python
from h2o.estimators import H2ONaiveBayesEstimator
```

Once you are done with the tutorial, remember to shut down the cluster.

``` python
h2o.cluster().shutdown()
```

## Next Steps

Regression Tutorial coming soon.




























H2O-3 Bugs Found & Reported

GLM plot giving error and not plotting

Jira link: https://0xdata.atlassian.net/browse/SW-1514

Plots killing kernel 
Jira link: https://0xdata.atlassian.net/browse/PUBDEV-6788

Tables not displayed properly
Jira link: https://0xdata.atlassian.net/browse/PUBDEV-6770


