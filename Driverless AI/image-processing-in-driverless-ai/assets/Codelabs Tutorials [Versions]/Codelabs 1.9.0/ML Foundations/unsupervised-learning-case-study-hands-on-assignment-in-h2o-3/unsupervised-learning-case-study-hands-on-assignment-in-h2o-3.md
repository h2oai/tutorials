---
id: unsupervised-learning-case-study-hands-on-assignment-in-h2o-3
summary: Check the list
categories: driverless
tags: automl, h2o-3, unsupervised
difficulty: 1
status: draft
feedback: https://github.com/h2oai/tutorials/issues

---

# Unsupervised Learning Case Study Hands-On Assignment in H2O-3

## Unsupervised Learning Case Study Hands-On Assignment in H2O-3 

### Objective

This is the hands-on exercise wherein you are required to complete a case study pertaining to Anomaly Detection with Isolation Forests using H2O-3.

For this assignment, you will be using the [credit card data set](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains information on various properties of credit card transactions. 

**Note: This tutorial has been built on Aquarium, which is H2O.ai's cloud environment providing software access for workshops, conferences, and training. The labs in Aquarium have datasets, experiments, projects, and other content preloaded. If you use your version of H2O-3 or Driverless AI, you will not see preloaded content.**
 
### Prerequisites

- Basic knowledge of Machine Learning and Statistics
- An [Aquarium](https://aquarium.h2o.ai/) Account to access H2O.ai’s software on the AWS Cloud. 
  - Need an [Aquarium](https://aquarium.h2o.ai/) account? Follow the instructions in the next section **Task 1 Create An Account & Log Into Aquarium** to create an account
  - Already have an Aquarium account? Log in and continue to **Task 2 Launch the H2O-3 & Sparkling Water Lab** to begin your exercise!
 
**Note: Aquarium's Driverless AI lab has a license key built-in, so you don't need to request one to use it. Each Driverless AI lab instance will be available to you for two hours, after which it will terminate. No work will be saved. If you need more time to further explore Driverless AI, you can always launch another lab instance or reach out to our sales team via the [contact us form](https://www.h2o.ai/company/contact/).**
 
### Task 1: Create An Account & Log Into Aquarium
 
Navigate to the following site: https://aquarium.h2o.ai/login and do the following: 

1. create a new account (if you don’t have one) 
2. log into the site with your credentials.
3. Navigate to the lab: H2O-3 and Sparkling Water Test Drive. Click on Start Lab and wait for your instance to be ready. Once your instance is ready, you will see the following screen.

![labs-urls](assets/labs-urls.jpg)

Click on the Jupyter url to start a jupyter notebook or the H2O Flow instance( if required). You can create a new Jupyter notebook and follow the steps defined below.
 
 
### Task 2: Open a New Jupyter Notebook

Open a new Jupyter Python3 Notebook by clicking New and selecting Python 3

![new-python-notebook](assets/new-python-notebook.jpg)

In this notebook, you will: 

1. Startup an H2O Cluster
2. Import necessary packages
3. Import the Credit Card dataset
4. Train an isolation forest
5. Inspect the Predictions


#### Deeper Dive and Resources:

- [Jupyter Notebook Tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)
 
### Task 3: Initialize the H2O-3 Cluster

In this section, you will use the `h2o.init()` method to initialize H2O. In the first cell of your notebook, you will:
 
1. Import the h2o python library 
2. Initialize the H2O cluster.  
3. Import the Isolation Forest Algorithm

You can enter the following in the first cell:

~~~python
import h2o
h2o.init()
from h2o.estimators import H2OIsolationForestEstimator
~~~
 
Your notebook should look like this:

![notebook](assets/notebook.jpg)

Then Run the cell to get started

![run-notebook](assets/run-notebook.jpg)
 
 
#### Deeper Dive and Resources:

- [Starting H2O from Python](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/starting-h2o.html#from-python)


### Task 4: Import the Credit Card Dataset 

We will be using the [credit card data set](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains information on various properties of credit card transactions. There are 492 fraudulent and 284,807 genuine transactions, which makes the target class highly imbalanced. We will not use the label during the anomaly detection modelling, but we will use it during the evaluation of our anomaly detection.

1.  Download the dataset from [here](https://www.kaggle.com/mlg-ulb/creditcardfraud) and then Enter the following in the next available cell and run it to bring in the credit card data. 

~~~python
#Import the dataset
df = h2o.import_file("creditcard.csv")
~~~

**Note:** The line with the # is a code comment.  These can be useful to describe what you are doing in a given section of code.

#### Deeper Dive and Resources:

- [Importing Data in H2O-3](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/importing-data.html)
 
### Task 5: Train the isolation forest model

There are multiple approaches to an unsupervised anomaly detection problem that try to exploit the differences between the properties of common and unique observations. The idea behind the Isolation Forest is as follows.

- We start by building multiple decision trees such that the trees isolate the observations in their leaves. Ideally, each leaf of the tree isolates exactly one observation from your data set. The trees are being split randomly. We assume that if one observation is similar to others in our data set, it will take more random splits to perfectly isolate this observation, as opposed to isolating an outlier.

- For an outlier that has some feature values significantly different from the other observations, randomly finding the split isolating it should not be too hard. As we build multiple isolation trees, hence the isolation forest, for each observation we can calculate the average number of splits across all the trees that isolate the observation. The average number of splits is then used as a score, where the less splits the observation needs, the more likely it is to be anomalous.

Now train your isolation forest. The last column (index 30) of the data contains the class label, so exclude it from the training process.

~~~python
seed = 12345 # For reproducability of the experiment
ntrees = 100 #Specify the number of Trees
isoforest = H2OIsolationForestEstimator( ntrees=ntrees, seed=seed)
# Specify x as a vector containing the names or indices of the predictor variables to use when building the model.
isoforest.train(x=df.col_names[0:30], training_frame=df)
~~~

#### Deeper Dive and Resources:

- [Isolation Forest in H2O-3(http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/if.html)


### Task 6: A look at the model’s predictions

Have a look at the predictions.

~~~python
predictions = isoforest.predict(df)

predictions
~~~

You will see that the `prediction h2o frame` contains two columns: `predict` showing a normalized anomaly score and `mean_length` showing the average number of splits across all trees to isolate the observation. These two columns should have the property of inverse proportion by their definition, as the less random splits you need to isolate the observation, the more anomalous it is. You can easily check that by :
 
~~~python 
predictions.cor()
~~~
 
### Task 7: Predicting Anomalies using Quantile

As we formulated this problem in an unsupervised fashion, how do we go from the average number of splits / anomaly score to the actual predictions? Using a threshold! If we have an idea about the relative number of outliers in our dataset, we can find the corresponding quantile value of the score and use it as a threshold for our predictions.

~~~python 
quantile = 0.95
quantile_frame = predictions.quantile([quantile])
quantile_frame
~~~

We can use the threshold to predict the anomalous class.

~~~python 
threshold = quantile_frame[0, "predictQuantiles"]
predictions["predicted_class"] = predictions["predict"] > threshold
predictions["class"] = df["Class"]
predictions
~~~

### Task 8: Evaluation

Because the isolation forest is an unsupervised method, it makes sense to have a look at the classification metrics that are not dependent on the prediction threshold and give an estimate of the quality of scoring. Two such metrics are Area Under the Receiver Operating Characteristic Curve (AUC) and Area under the Precision-Recall Curve (AUCPR).

`AUC` is a metric evaluating how well a binary classification model distinguishes true positives from false positives. The perfect AUC score is 1; the baseline score of a random guessing is 0.5.

`AUCPR` is a metric evaluating the precision recall trade-off of a binary classification using different thresholds of the continuous prediction score. The perfect AUCPR score is 1; the baseline score is the relative count of the positive class.

For highly imbalanced data, AUCPR is recommended over AUC as the AUCPR is more sensitive to True positives, False positives and False negatives, while not caring about True negatives, which in large quantity usually overshadow the effect of other metrics.

~~~python
%matplotlib inline
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

 
def get_auc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score
 
 
def get_aucpr(labels, scores):
    precision, recall, th = precision_recall_curve(labels, scores)
    aucpr_score = np.trapz(recall, precision)
    return precision, recall, aucpr_score
 
 
def plot_metric(ax, x, y, x_label, y_label, plot_label, style="-"):
    ax.plot(x, y, style, label=plot_label)
    ax.legend()
    
    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)
 
 
def prediction_summary(labels, predicted_score, predicted_class, info, plot_baseline=True, axes=None):
    if axes is None:
        axes = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]
 
    fpr, tpr, auc_score = get_auc(labels, predicted_score)
    plot_metric(axes[0], fpr, tpr, "False positive rate",
                "True positive rate", "{} AUC = {:.4f}".format(info, auc_score))
    if plot_baseline:
        plot_metric(axes[0], [0, 1], [0, 1], "False positive rate",
                "True positive rate", "baseline AUC = 0.5", "r--")
 
    precision, recall, aucpr_score = get_aucpr(labels, predicted_score)
    plot_metric(axes[1], recall, precision, "Recall",
                "Precision", "{} AUCPR = {:.4f}".format(info, aucpr_score))
    if plot_baseline:
        thr = sum(labels)/len(labels)
        plot_metric(axes[1], [0, 1], [thr, thr], "Recall",
                "Precision", "baseline AUCPR = {:.4f}".format(thr), "r--")
 
    plt.show()
    return axes
 
 
def figure():
    fig_size = 4.5
    f = plt.figure()
    f.set_figheight(fig_size)
    f.set_figwidth(fig_size*2)
 
 
h2o_predictions = predictions.as_data_frame()
 
figure()
axes = prediction_summary(
    h2o_predictions["class"], h2o_predictions["predict"], h2o_predictions["predicted_class"], "h2o")

~~~

Code link: https://gist.github.com/parulnith/48649e0c82dbb59c6f36e7a507fa1eef


### Next Steps

In the above study, you  learned about the isolation forests, their underlying principle, how to apply them for unsupervised anomaly detection, and how to evaluate the quality of anomaly detection once we have corresponding labels .You can now proceed on to attempt the **Quiz 4: Unsupervised Machine Learning with H2O-3.**