# Disparate Impact Analysis  

## Outline
- [Objective](#objective)
- [Prerequisites](#prerequisites)
- [Task 1: Launch Machine Learning Interpretability Experiment](#task-1-launch-machine-learning-interpretability-experiment) 
- [Task 2: Concepts](#task-2-concepts)
- [Task 3: Disparate Impact Analysis](#task-3-disparate-impact-analysis)
- [Task 4: Disparate Impact Analysis](#task-4-disparate-impact-analysis)
- [Task 5: Sensitivity Analysis Part 1: Checking for Bias](#task-5-sensitivity-analysis-part-1-checking-for-bias)
- [Task 6: Sensitivity Analysis Part 2: Checking for Bias](#task-6-sensitivity-analysis-part-2-checking-for-bias)
- [Next Steps](#next-steps)

## Objective

As firms use ML to help them around credit/loan-decisions, cross-sell promotions, and determine the next best action, they must know how certain customer features are being weighed into the ML models in production. Further, they are also required to understand whether the ML models are not negatively impacting protected classes of customers or unfairly weighting for these types of classes. A lack of understanding of an ML Model's ins of production can lead to legal and financial risks when discovering that the ML model in production discriminates (bias) against certain ethnicities, genders, etc.

Additionally, as firms have looked to leverage AI to make more and more decisions for the company, the discussion of Human-Centered Machine learning has become increasingly important. Data science practitioners and firms deploying AI in production want to 'get under the hood' of their models to see what impacts decisions. Hence, in this tutorial, we will build an AI model predicting whether someone will be defaulting on their next credit card payment. Right after, we will use the following two Driverless AI features to analyze and check for fairness. 

- Disparate Impact Analysis (DIA)
- Sensitivity Analysis(SA)

As a matter of speaking, the above two features provide a solution to a common problem in ML: the multiplicity of good models.  It is well understood that for the same set of input features and prediction targets, complex machine learning algorithms can produce multiple accurate models with very similar, but not the same, internal architectures: the multiplicity of good models [1]. This alone is an obstacle to interpretation, but when using these types of tools as interpretation tools or with interpretation tools, it is important to remember that details of explanations can change across multiple accurate models. This instability of explanations is a driving factor behind the presentation of multiple explanatory results in Driverless AI, enabling users to find explanatory information that is consistent across multiple modeling and interpretation techniques. And such explanatory results can be accessed by the **Disparate Impact Analysis** and **Sensitivity Analysis(SA)** features/tools. 

With the above in mind, let us discover how we can better understand our models. 

### References 

- [1] [Jerome Friedman, Trevor Hastie, and Robert Tibshirani. The Elements of Statistical Learning. Springer, New York, 2001.](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf)


## Prerequisites

You will need the following to be able to do this tutorial:

- Basic knowledge of Machine Learning and Statistics
- Basic knowledge of Driverless AI or doing the [Automatic Machine Learning Introduction with Driverless AI](https://training.h2o.ai/products/tutorial-1a-automatic-machine-learning-introduction-with-driverless-ai) 
- Completion of the following tutorial: [Machine Learning Interpretability](https://training.h2o.ai/products/tutorial-1c-machine-learning-interpretability-tutorial)
- A **Two-Hour Test Drive session**: Test Drive is H2O.ai's Driverless AI on the AWS Cloud. No need to download software. Explore all the features and benefits of the H2O Automatic Learning Platform.
  - Need a **Two-Hour Test Drive** session?Follow the instructions on [this quick tutorial](https://training.h2o.ai/products/tutorial-0-getting-started-with-driverless-ai-test-drive) to get a Test Drive session started. 

**Note:  Aquarium’s Driverless AI Test Drive lab has a license key built-in, so you don’t need to request one to use it. Each Driverless AI Test Drive instance will be available to you for two hours, after which it will terminate. No work will be saved. If you need more time to further explore Driverless AI, you can always launch another Test Drive instance or reach out to our sales team via the [contact us form](https://www.h2o.ai/company/contact/).**

## Task 1: Launch Machine Learning Interpretability Experiment

### About the Dataset

For this exercise, we will use the same credit card default prediction dataset that we used in the first MLI tutorial. This dataset contains information about credit card clients in Taiwan from April 2005 to September 2005. Features include demographic factors, repayment statuses, history of payment, bill statements, and default payments. The data set comes from the [UCI Machine Learning Repository: UCI_Credit_Card.csv](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#) And this dataset has a total of 25 Features(columns) and 30,000 clients(rows).

### Download Dataset

When looking at the **UCI_Credit_Card.csv**, we can observe that column **PAY_0** was suppose to be named **PAY_1**. Accordingly, we will solve this problem using a data recipe that will change the column's name to **PAY_1**. The data recipe has already been written and can be found [here](https://s3.amazonaws.com/data.h2o.ai/DAI-Tutorials/MLI+Tutorials/uci_credit_card_recipe.py). Download the data recipe and name it ```uci_credit_card_recipe.py```. Make sure it's saved as a **.py** file. 
 
Now upload the data recipe to the Driverless AI dataset's page. In the **DATASETS** page click **+ ADD DATASET(OR DRAG & DROP)** and select **UPLOAD DATA RECIPE**: 

![upload-data-recipe](assets/upload-data-recipe.jpg)

After it imports successfully, you will see the following CSV on the **DATASETS** page: **UCI_Credit_Card.csv**. Click on the ``UCI_Credit_Card.csv`` file then select **Details**:

![launch-experiment-variable-details](assets/launch-experiment-variable-details.jpg)

Review the columns in the dataset and pay attention to the specific attributes we will want to keep an eye on, such as **SEX**, **EDUCATION**, **MARRIAGE**, and **AGE**. Note that these demographic factors in predicting default credit payments.

- When we think about disparate impact, we want to analyze whether specific classes are being treated unfairly. For example, single/non-college educated clients.


Recall the dataset metrics: 

- **ID** - Row identifier (which will not be used for this experiment)
-  **LIMIT_BAL** - Amount of the given credit: it includes the individual consumer credit and family (supplementary) credit
- **Sex** - Gender (1 =  male; 2 = female)
- **EDUCATION**- Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)
- **MARRIAGE** - Marital status (1 = married; 2 = single; 3 = others)
- **Age**
- **PAY_1 - PAY_6**: History of past payment:
  - -2: No consumption
  - -1: Paid in full 
  - 0: The use of revolving credit
  - 1 = payment delay for one month
  - 2 = payment delay for two months; . . .; 
  - 6 = payment delay for six months
-  **BILL_AMT1 - BILL_AMT6** - Amount of bill statement 
-  **PAY_AMT1 -PAY_AMT6** - Amount of previous payment 
-  **default.payment.next.month** - Default (1: Yes, 0: No)

Return to the Datasets page.

Click on the ``UCI_Credit_Card.csv`` file then select **Predict**. Select Not Now on the "First time Driverless AI" dialog box.
- Name your experiment: ```UCI_CC MLI```
- Select the following feature as a Target Column: **default.payment.next.month**


At this point, your experiment preview page should look as follows:

![experiment-pre-view-page](assets/experiment-pre-view-page.jpg)



Usually, when predicting default, we will drop attributes such as **SEX**, **EDUCATION**, **MARRIAGE**, **AGE**, and **LIMIT_BALL**. Such features are drop because they shouldn't be considered because of conscious and unconscious bias. Not using such features will prevent decisions from being based on uncontrollable features such as sex. But what if we had no idea a given feature had or could lead to bias and unfairness. How can we find out that using a given feature leads to bias? The answer can be found when analyzing predictions using the Disparate Impact Analysis and Sensitivity Analysis tools. Therefore, we will not drop any columns for now, and let's discover how Driverless AI can perhaps highlight bias features. Not dropping any columns will allow us to understand how we can conclude that a feature is biased when it's not clear that a feature will generate bias when used on an ML model. The idea here is that when we, later on, analyze a feature's level of impact on single or overall predictions, we can decide whether that given feature in question is generating unfair predictions. As a result, we can drop the features found to be biased and, at last, rebuilt a model that is bias-free. 

Again, we will assume that we have no idea that features such as **SEX** can lead to possible unfair predictions when used on particular ML models.  

In the **TRAINING SETTINGS** as shown below: it is essential to make sure the **Interpretability** setting is at **7**. On the left-hand side, verify that **Monotonicity Constraints** is enabled. Enabling **Monotonicity Constraints** is important to Disparate Impact Analysis. If we use an unconstrained model and group fairness metrics, we risk creating group fairness metrics that appear to be reasonable. The consequence of creating group fairness metrics that appear to be reasonable is the illusion that individuals within that group may NOT be treated differently or unfairly. The local (individual) discrimination would likely not appear or be visible in that group metric.

![launch-experiment-interpretability-settings](assets/launch-experiment-interpretability-settings.jpg)

Now we will adjust some settings to our experiment. We will make use of the **Expert Settings** feature to make these adjustments. As discussed in previous tutorials, Driverless AI provides various options in the Expert Settings that let you customize your experiment. Now, click on the **Expert Settings** option, located on the top right corner of the **TRAINING SETTINGS** option: 

- In the **EXPERT SETTINGS**, select the model tab, and adjust the settings to create a single *XGBoost GBM Model*: 
  - Turn off all the models besides the **XGBoost GBM Models** setting.
    > XGBoost is a type of gradient boosting method that has been widely successful in recent years due to its good regularization techniques and high accuracy. This is set to Auto by default. If enabled, XGBoost models will become part of the experiment (for both the feature engineering part and the final model).
  - Then scroll down and adjust the **Ensemble level for final modeling pipeline** setting to 0 for this exercise's interpretability purposes. 
  - Click **Save** and return to the experiment preview page.

  ![launch-experiment-model-settings](assets/launch-experiment-model-settings.jpg)
  ![launch-experiment-ensemble-level](assets/launch-experiment-ensemble-level.jpg)

The last step here is to click **REPRODUCIBLE**, then run the experiment:

![launch-experiment-reproducible](assets/launch-experiment-reproducible.jpg)

While the experiment runs, let's go over a few concepts that will be crucial when conducting a **Disparate Impact** and **Sensitivity** analysis. 

## Task 2: Concepts

### Fairness & Bias

Fairness in Machine Learning & AI has been a critical focus for many practitioners and industries. The goal at its core is quite simple: ensure your models are not treating one population or group worse than another. However, upon further review, this task becomes more complicated to verify because fairness is not a term with an agreed-upon legal definition. 

In colloquial terms, bias tends to imply that a person has a slightly incorrect or exaggerated opinion on a subject matter based on their personal experience, whether or not that represents the truth. Frequently the term will be used this way in Machine Learning as well. However, it is important to understand bias in a statistical term with a different meaning as well. 

‘In statistics, the **bias** (or **bias function**) of an estimator is the difference between this estimator's expected value and the true value of the parameter being estimated. An estimator or decision rule with zero bias is called **unbiased**. In statistics, "bias" is an **objective** property of an estimator.’[1]

More specifically, to Machine Learning, there is a concept called bias-variance tradeoff that appears often:


<p align="center"> 
    <img src='assets/bias-variance-tradeoff.jpg' width="500"></img>    
</p>

<p align="center"> 
    <img src='assets/model-complexity.jpg' width="500"></img>    
</p>

- Bias is the simplifying assumptions made by the model to make the target function easier to approximate.[2]
- Variance is the amount that the estimate of the target function will change given different training data.[2]
- The goal of any supervised machine learning algorithm is to achieve low bias and low variance. In turn the algorithm should achieve good prediction performance.[2]
  - Linear machine learning algorithms often have a high bias but a low variance.[2]
  - Nonlinear machine learning algorithms often have a low bias but a high variance.[2]
- The parameterization of machine learning algorithms is often a battle to balance out bias and variance.The goal of any supervised machine learning algorithm is to achieve low bias and low variance. In turn the algorithm should achieve good prediction performance.[2]
  - Therefore, trade-off is tension between the error introduced by the bias and the variance.[2]

**Note**: If the theory of fairness and ethics in AI interests you, we have listed some of our favorite resources below on the topic that dives much deeper.

### Disparate Impact Analysis

In most law jurisdictions,  **Disparate Impact** refers to the conscious and unconscious practices adversely impacting one group of people of a protected characteristic more than another. Such characteristics constitute someone's race, color, religion, national origin, sex, and disability status.

When the discussion of ‘fairness’ or ‘ethical AI’ comes up, one of the best possible methodologies for vetting fairness is Disparate Impact Analysis. Disparate Impact Analysis or DIA, which is sometimes called Adverse Impact Analysis, is a way to measure quantitatively the adverse treatment of protected classes, which leads to discrimination in hiring, housing, etc., or in general, any public policy decisions. The regulatory agencies will generally regard a selection rate for any group with less than four-fifths (4/5) or eighty percent of the rate for the group with the highest selection rate as constituting evidence of adverse impact.

### Sensitivity Analysis/What-If Analysis

Sensitivity analysis, sometimes called what-if analysis is a mainstay of model debugging. It’s a very simple and powerful idea: simulate data that you find interesting and see what a model predicts for that data. Because ML models can react in very surprising ways to data they’ve never seen before, it’s safest to test all of your ML models with sensitivity analysis.

**Sensitivity analysis** is the study of how the [uncertainty](https://en.wikipedia.org/wiki/Uncertainty) in the output of a [mathematical model](https://en.wikipedia.org/wiki/Mathematical_model) or system (numerical or otherwise) can be divided and allocated to different sources of uncertainty in its inputs.[3] [4] 

A related practice is [uncertainty analysis](https://en.wikipedia.org/wiki/Uncertainty_analysis), which has a greater focus on [uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification) and [propagation of uncertainty](https://en.wikipedia.org/wiki/Propagation_of_uncertainty); ideally, uncertainty and sensitivity analysis should be run in tandem.

One of the simplest and most common approaches is that of changing one-factor-at-a-time (OAT), to see what effect this produces on the output.[5] [6] [7] OAT customarily involves:

- Moving one input variable, keeping others at their baseline (nominal) values, then,
- Returning the variable to its nominal value, then repeating for each of the other inputs in the same way.

Sensitivity may then be measured by monitoring changes in the output, e.g. by [partial derivatives](https://en.wikipedia.org/wiki/Partial_derivatives) or [linear regression](https://en.wikipedia.org/wiki/Linear_regression). 


### Confusion Matrices

It is effortless to assume that the goal is accuracy when using machine learning: the percentage of your predictions being correct. While accuracy can be a useful metric of success, it is often dubious. Let’s build on a very relevant industry example: 

Fraud (Anomaly Detection): Let’s assume we are dealing with a 100,000-row dataset where we know there is some small amount of fraud; let’s say 10. If accuracy is our benchmark, then your model will predict “Not-Fraud” every time, and the accuracy will be 99.99%, but you have failed to identify any instances of fraud. These cases focus on what in a confusion matrix is defined as True Positives (TP):


<p align="center"> 
    <img src='assets/confusion-matrices-1.jpg' width="500"></img>    
</p>

In the case of identifying fraud, you would almost always prefer a prediction table like this, to make sure you can correctly identify fraud instances as they occur:

<p align="center"> 
    <img src='assets/confusion-matrices-2.jpg' width="500"></img>    
</p>


Now every case is different, and often in business environments, there exist certain cost functions associated with false negatives and false positives, so it is essential to be aware that every case has many considerations. We want to provide a few of the key metrics associated with confusion matrices that come up in the industry, depending on the problem you are trying to solve.

- **Sensitivity, Recall, Hit Rate, True Positive Rate**:
  - True Positive Rate = True Positive / (True Positive + False Negative)
- **Specificity, Selectivity, True Negative Rate**:
  - True Negative Rate = True Negative / (True Negative + False Positive)
- **Precision, Positive Predictive Value**:
  - Precision = True Positives / (True Positive + False Positive)

With this context in mind, let’s move forward and dive into the experiment!


### References

- [1] [Bias of an estimator](https://en.wikipedia.org/wiki/Bias_of_an_estimator)
- [2] Jason Brownlee PhD . ["Gentle Introduction to the Bias-Variance Trade-Off in Machine Learning"](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/). Machine Learning Mastery. October 25, 2019. 
- [3]  Saltelli, A. (2002). ["Sensitivity Analysis for Importance Assessment"](https://en.wikipedia.org/wiki/Sensitivity_analysis#cite_note-Risk_Analysis-1). Risk Analysis. 22 (3): 1–12. CiteSeerX 10.1.1.194.7359. doi:10.1111/0272-4332.00040. PMID 12088235.
- [4]  Saltelli, A.; Ratto, M.; Andres, T.; Campolongo, F.; Cariboni, J.; Gatelli, D.; Saisana, M.; Tarantola, S. (2008). [Global Sensitivity Analysis: The Primer. John Wiley & Sons](https://en.wikipedia.org/wiki/Sensitivity_analysis#cite_note-Primer-2).
- [5] [Sensitivity analysis](https://en.wikipedia.org/wiki/Sensitivity_analysis#cite_note-15)
- [6]  Leamer, Edward E. (1983). ["Let's Take the Con Out of Econometrics"](https://en.wikipedia.org/wiki/Sensitivity_analysis#cite_note-16). American Economic Review. 73 (1): 31–43. JSTOR 1803924.
- [7]  Leamer, Edward E. (1985). ["Sensitivity Analyses Would Help". American Economic Review](https://en.wikipedia.org/wiki/Sensitivity_analysis#cite_note-17). 75 (3): 308–313. JSTOR 1814801.


### Deeper Dive and Resources

- [Mitigating Bias in AI/ML Models with Disparate Impact Analysis …](https://medium.com/@kguruswamy_37814/mitigating-bias-in-ai-ml-models-with-disparate-impact-analysis-9920212ee01c)
- [In Fair Housing Act Case, Supreme Court Backs 'Disparate Impact' Claims](https://www.npr.org/sections/thetwo-way/2015/06/25/417433460/in-fair-housing-act-case-supreme-court-backs-disparate-impact-claims)
- [Fairness and machine learning](https://fairmlbook.org/)
- [50 Years of Test (Un)fairness: Lessons for Machine Learning](https://arxiv.org/pdf/1811.10104.pdf)
- [Biased Algorithms Are Easier to Fix Than Biased People](https://www.nytimes.com/2019/12/06/business/algorithm-bias-fix.html)
- [Discrimination in the Age of Algorithms](https://arxiv.org/abs/1902.03731)
- [Understanding and Reducing Bias in Machine Learning](https://towardsdatascience.com/understanding-and-reducing-bias-in-machine-learning-6565e23900ac)


## Task 3: Disparate Impact Analysis


By now your experiment should be done. Since we are already familiar with the experiment dashboard let’s move forward by selecting the **INTERPRET THIS MODEL** option. If you need to review the Driverless AI UI please refer back to this tutorial: [Automatic Machine Learning Introduction with Driverless AI.](https://training.h2o.ai/products/tutorial-1a-automatic-machine-learning-introduction-with-driverless-ai) 

![experiment-complete](assets/experiment-complete.jpg)


After the model is interpreted, you will be taken to the "MLI: Regression and Classification Explanations" page. The *DAI Model* tab (not to be confused with the **DIA** Metrics: Disparate Impact Analysis Metrics) should already be selected for you thereafter, click on *Disparate Impact Analysis* located at the bottom left corner of the page. The following will appear: 

![dia](assets/dia.jpg)

**Note**: This plot is available for binary classification and regression models.

DIA is a technique that is used to evaluate fairness. Bias can be introduced to models during the process of collecting, processing, and labeling data—as a result, it is important to determine whether a model is harming certain users by making a significant number of biased decisions.

DIA typically works by comparing aggregate measurements of unprivileged groups to a privileged group. For instance, the proportion of the unprivileged group that receives the potentially harmful outcome is divided by the proportion of the privileged group that receives the same outcome—the resulting proportion is then used to determine whether the model is biased. Refer to the Summary section to determine if a categorical level  is fair in comparison to the specified reference level and user-defined thresholds. Fairness All is a true or false value that is only true if every category is fair in comparison to the reference level.

Disparate impact testing is best suited for use with constrained models in Driverless AI, such as linear models, monotonic GBMs, or RuleFit. The average group metrics reported in most cases by DIA may miss cases of local discrimination, especially with complex, unconstrained models that can treat individuals very differently based on small changes in their data attributes.

- **Note**: We only enabled a XGBoost GBM Model and we constrained the model by setting the interpretability nob to >= 7. 

DIA allows you to specify a disparate impact variable (the group variable that is analyzed), a reference level (the group level that other groups are compared to), and user-defined thresholds for disparity. Several tables are provided as part of the analysis:

- **Group metrics**: The aggregated metrics calculated per group. For example, true positive rates per group.
- **Group disparity**: This is calculated by dividing the ```metric_for_group``` by the ``reference_group_metric``. Disparity is observed if this value falls outside of the user-defined thresholds.
- **Group parity**: This builds on Group disparity by converting the above calculation to a true or false value by applying the user-defined thresholds to the disparity values.

In accordance with the established four-fifths rule, user-defined thresholds are set to 0.8 and 1.25 by default. These thresholds will generally detect if the model is (on average) treating the non-reference group 20% more or less favorably than the reference group. Users are encouraged to set the user-defined thresholds to align with their organization’s guidance on fairness thresholds.

### Metrics - Binary Classification

The following are formulas for error metrics and parity checks utilized by binary DIA. Note that in the tables below:

 **tp**= true positive **|** **fp** = false positive **|** **tn** = true negative **|** **fn** = false negative

<p align="center"> 
    <img src='assets/parity-check-one.jpg' width="500"></img>    
</p>
<p align="center"> 
    <img src='assets/parity-check-two.jpg' width="500"></img>    
</p>

Recall that our current experiment is a classification model, and therefore, the above metrics will be generated by the Disparate Impact Analysis(DIA) tool. In contrast, if it were a regression problem, we will see different metrics. To learn about metrics DIA will generate for a regression experiment, click [here](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/interpret-understanding.html#metrics-regression).

**Note**:

- Although the process of DIA is the same for both classification and regression experiments, the returned information is dependent on the type of experiment being interpreted. An analysis of a regression experiment returns an actual vs. predicted plot, while an analysis of a binary classification experiment returns confusion matrices.
- Users are encouraged to consider the explanation dashboard to understand and augment results from disparate impact analysis. In addition to its established use as a fairness tool, users may want to consider disparate impact for broader model debugging purposes. For example, users can analyze the supplied confusion matrices and group metrics for important, non-demographic features in the Driverless AI model.

With the above in mind, let's continue with our experiment analysis: 

Observe the Global Confusion Matrix for our experiment:

![dia-2](assets/dia-2.jpg)

With the above in mind, you can look at the Global Confusion Metrix of our experiment:


- The confusion matrix above is structure as follows: 

  ![confusion-matrices-explanation](assets/confusion-matrices-explanation.jpg)

  - **True Positive Rate** = 3091/(3091 + 3545) = 0.4657
    - probability that an actual positive will test positive
  - **True Negative Rate** = 21301/(21301 + 2063) = 0.911
    - which is the probability that an actual negative will test negative
  - **False Negatives** = 3545.0
  - **False Positive** = 2063.0
    - Below we will analyze whether members of a protective class are not unfairly labeled as false positives (wrong prediction of default)


**Note**: The confusion matrix is base on the **MAXIMIZED METRIC** and **CUT OFF** value found on the summary section: 
- Predictive models often produce probabilities, not decisions. So to make a decision with a model-generated predicted probability for any one client, we need a numeric cutoff which we say a client will default and below which we say they will not default. Cutoffs play a crucial role in DIA as they impact the underlying measurements used to calculate diparity. In fact, tuning cutoffs carefully is a potential remdiation tactic for any discovered disparity. There are many accepted ways to select a cutoff (besides simply using 0.5) and in this experiment Driverless AI has selected a balance between the model's recall (true positive rate) and it's precision using the **F1** statistic. Using precision and recall to select a cutoff is sometimes seen as more robust to imbalanced data than the standard ROC approach. Maximizing **F1** typically results in a good balance between sensitivity and precision. In this experiment, Driverless AI has decided that the best **CUT OFF** value is **0.56**, leading to a maximization of precision and recall. 
- In other words, a Driverless AI model will return probabilities, not predicted classes. To convert probabilities to predicted classes, a cutoff needs to be defined. Driverless AI iterates over possible cutoffs to calculate a confusion matrix for each cutoff. It does this to find the maximum F metric value. Driverless AI’s goal is to continue increasing this maximum F metric.
- The F1 score provides a measure for how well a binary classifier can classify positive cases (using a threshold value). The F1 score is calculated from the harmonic mean of the precision and recall. An F1 score of 1 means both precision and recall are perfect and the model correctly identified all the positive cases and didn’t mark a negative case as a positive case. If either precision or recall are very low it will be reflected with a F1 score closer to 0.
  - F1 equation: F1 = 2((Precision)(Recall)/Precision + Recall)
  - Where:
    - **precision** is the positive observations (true positives) the model correctly identified from all the observations it labeled as positive (the true positives + the false positives).
    - **recall** is the positive observations (true positives) the model correctly identified from all the actual positive cases (the true positives + the false negatives).

In the following task, let's put the above matrics into perspective in terms, of whether unfairness is present. 

## Task 4: Group Disparity and Parity

To begin our analysis on each feature (column) and determine whether it holds a disparate impact, let's consider the above steps on how it could be determined using DIA. 

### Feature: SEX

First, let us observe the **SEX** feature: 

On the top right corner, click on the **Disparate Impact Variable** button. Select the **SEX** variable: 

![disparate-impact-select](assets/disparate-impact-select.jpg)

On the **Reference Level** button and change the value from **2(Female)** to **1(Male)**: 

![reference-level](assets/reference-level.jpg)

The following will appear: 

![dia-fairness-one](assets/dia-fairness-one.jpg)
![dai-fairness-two](assets/dia-fairness-two.jpg)



Make sure the **Reference level** is toggled to 1(Male). With DIA the reference level is somewhat ambiguous, but generally, we want to set it to the category or population we believe may be receiving better treatment compared to other classes. 


From a general observation we can see that base on the **Summary** section: 

- It looks like the model didn't do well with respect to eliminating bias around Gender! The FAIRNESS 1 (Male) is True, FAIRNESS 2 (Female) is False, and the FAIRNESS ALL is False that tells us the model accuracy is only fair to males  — definitely breaking the four-fifth rule.
  - **When conducting a DIA analysis, always refer to the Summary section to determine if a categorical level is fair in comparison to the specified reference level and user-defined thresholds. Fairness All is a true or false value that is only true if every category is fair in comparison to the reference level.**
- If we had to break this down, The Disparate Impact Analysis basically took model prediction results which was from 11K males, 18k females and looked at various measures such as Accuracy, Adverse Impact, True Positive Rate, Precision, Recall, etc., across both binary classes and then found the ratios are not comparable across the two groups over the desired cutoff value. 
  - A reminder *Adverse Impact* refers to the following definition: 
    - Adverse impact is the negative effect an unfair and biased selection procedure has on a protected class. It occurs when a protected group is discriminated against during a selection process, like a hiring or promotion decision.[8]  
      - In the US, protected classes include race, sex, age (40 and over), religion, disability status, and veteran status.
- Further, *True Positive Rate* refers to the following definition: 
  - In machine learning, the true positive rate, also referred to sensitivity or recall, is used to measure the percentage of actual positives which are correctly identified.[9]
- Further we also see a slight population imbalance but not at a worrisome level yet. It will be a problem when the imbalance is huge because it will mean that the model will be learning (through examples) from a particular population or group. Though it could be the case, this slight population imbalance is causing unfairness, but further analysis is required to support that conclusion.  

Scroll down and let's take a look at *Group Disparity*.

![0.8-low-threshold](assets/0.8-low-threshold.jpg)

You will notice that your benchmark (1/Male) should be one as it provides the level to compare. In financial services, there is an ‘accepted’ rule of thumb that, as a benchmark, one class should not be treated 80% less favorably than another, as a starting point. This benchmark is up for debate, and therefore, we will see how this will develop going forward, but for now, we have set the low (unfairness) threshold to *.8** (it should already be set to .8). As a result, with a .8 low threshold, we can see that none of the group disparity metrics are less than point 8, which is a good sign.

5\. Now let’s adjust the *low threshold* to .9 and see what happens: 

![0.9-low-threshold](assets/0.9-low-threshold.jpg)

We can see that if we adjust the low threshold, the group disparity metrics will become highlighted and begin to flag, saying that if .9 is the cutoff,  this class (2/Women) will be treated unfairly by the model. Note: the *True Positive Rate Disparity* and *False Negative Rate Disparity* are not affected.

6\. Let’s scroll down and investigate *Group Parity* and check if all classes are being treated fairly by the model under the benchmark thresholds.

![group-parity](assets/group-parity.jpg)

Here we can see *True* across all classes and metrics, which is what we want to see.

### References

[8] [WHAT IS ADVERSE IMPACT? AND WHY MEASURING IT MATTERS](https://www.hirevue.com/blog/what-is-adverse-impact-and-why-measuring-it-matters)

[9] [Encyclopedia of Systems Biology](https://link.springer.com/referenceworkentry/10.1007%2F978-1-4419-9863-7_255)

## Task 5: Sensitivity Analysis Part 1: Checking for Bias

1\. Let’s start up a new experiment with the same dataset as before. Keep the settings & target variable the same; however, this time, let’s keep all the columns in the dataset. 

![new-experiment](assets/new-experiment.jpg)

After the experiment is over:

 - Click on *INTERPRET THIS MODEL*

 - After, in the *DAI Models* tab you should click on the Sensitivity Analysis option 

After that, you should land on our Sensitivity Analysis Dashboard: 

![sa-ui](assets/sa-ui.jpg)

Some things to notice:

 1. In our *Summary* information for the dataset located on the left side of the dashboard, we, in particular, can see our chosen cutoff metric, and the number for that metric. 

![sa-ui-summary](assets/sa-ui-summary.jpg)

 - In our *Summary* information for the dataset located on the left side of the dashboard, we, in particular, can see our chosen cutoff metric, and the number for that metric. Our *CUTOFF* is 0.2676... Anything below the *CUTOFF* will mean the model predicts a customer will not default, while anyone greater than or equal to the *CUTOFF* will default. 

2\. This pink summary locator represents the “Average” customer in the dataset, i.e., the average of all computable variables.  

![sa-ui-threshold](assets/sa-ui-threshold.jpg)

- The *Current Working Set Score* indicates that the mean score prediction is .24060 and that the most common prediction is False, which makes sense.

3\. Here we can choose to filter down on various portions of the confusion matrix and review each row and prediction.

![sa-ui-cm-table](assets/sa-ui-cm-table.jpg)

4\.  Now that we have familiarized ourselves with the UI let’s experiment! Reminder *Sensitivity Analysis* enables us to tinker with various settings in the data to see if certain features affect the outcome when we know that they should not. Let’s start by adjusting an entire feature.

 - If you remember from the previous exercise, the feature PAY_0 was extremely important; if not, you can jump back to *Transformed Shapley* and double-check. You can find the *Transformed Shapley*  in the *DAI MODEL* tab. 

![shapley](assets/shapley.jpg)

5\. You can also check the *Partial Dependence Plot* and see the probability of defaulting increases when PAY_0 is two months late. 

![partial-dependence-plot-of-pay_0](assets/partial-dependence-plot-of-pay_0.jpg)

Now that we know that being two months late on PAY_0 is terrible and knowing that the average mean score prediction is *0.24060*, what will occur if we were to set all customers to have PAY_0=2? Will the average mean score prediction increase or decrease? 

To set all customers PAY_0 to 2, please consider the following steps: 

 1. Click on top of the PAY_0 variable. 

![change-pay-0](assets/change-pay-0.jpg)

 2. A box will appear to make sure the absolute radio button is selected. Set the *Absolute* to 2. After, click *Set*.

![pay-0-equal-to-2](assets/pay-0-equal-to-2.jpg)

 3. Click the *RESCORE* button. 

![rescore](assets/rescore.jpg)

6\. We can check “Current Working Set Score” on the right to see a summation of what occurred. In this run, we see that by switching all to PAY_0=2 that we over doubled the average mean, implying that our model would reject significantly more of the credits because the perceived probability of default was much higher. Our current score is now *0.46028* from a prior score of *0.24060*. Consequently, the absolute change is of *0.21968* an increase of *91.31%* change.

![delta](assets/delta.jpg)


7\. Let's take this one step further and try another adjustment. Let's adjust PAY_AMT2 to 80% of what it originally was i.e., Let's see what happens to the model when we make our entire population only pay 80% of what they did. As you may notice, the variable *PAY_AMT2* is not on the table, but don't worry, we can add it by following these quick steps: 

 1. Click on the *Plus* icon, and a box will appear. 

  ![add-option](assets/add-option.jpg)

 2. Check the *PAY_AMT2* variable and click *SET*

 3. Right after, similar to how we change the value of *PAY_0*, we will click on the *PAY_AMT2* variable. 

 4. Select the *Percentage* radio button and set the *Percentage* to 80. Click *SET*. 

 ![pay2-80-percent](assets/pay2-80-percent.jpg)

 5. Click the *RESCORE* button. 

    Interestingly enough, when we finish rescoring and review results, we discover that this adjustment had virtually no effect whatsoever. In this case, the absolute change when modifying *PAY_AMT2* is a *0.00074* increase. It is a crucial consideration in Machine Learning to be aware that there will be many situations where you might think a variable would be important where it is not and vice versa.

![no-change](assets/no-change.jpg)

6. Let's inspect this from a different angle, but before, let's restore the original *Sensitivity Analysis* results. Click the reset button located on the bottom right corner of the *Sensitivity Analysis* graph.  

![reset](assets/reset.jpg)

## Task 6: Sensitivity Analysis Part 2: Checking for Bias

 For this subsequent analysis, we will tinker with an individual user and see what attribute changes might push them over or under the default/approval threshold. There are a few ways to do this, but since we know the cutoff is *0.26765209436416626*, we will try to find  a particular customer (blue or yellow circle) very close to the cutoff. Note, there are several ways we can filter to get close to a person close to the cutoff. In this case, I was able to find someone close to the cutoff line by filtering as follows: 

On the left side of the table (at the bottom of the page), you will be able to locate the filter options. In my case, I selected the **ID** variable and filter by *ID < 4*. 

![filter](assets/filter.jpg)

 Once you get a narrow range, you should see up close which customers are the closest predictions to the cutoff line, as shown above. In this case, we will experiment with a customer holding the *ID* number 3. 

![before-values-change](assets/before-values-change.jpg)

Now let’s see if we can independently adjust some of the demographic features and push customer three over the threshold from negative(not predicted as defaulting) to positive(predicted to default). Since we are discussing fairness and regulatory considerations, let’s select just the demographic variables to learn more. To choose the demographic variables such as **AGE**, **SEX**, **EDUCATION**, and **MARRIAGE** please follow the following steps: 

 1. Click the plus icon 

 2. Check the radio buttons for **AGE**, **SEX**, **EDUCATION**, and **MARRIAGE** 

 3. Click *SET*

After following the above steps, scroll to the right to see the newly added columns name: **AGE**, **SEX**, **EDUCATION**, and **MARRIAGE**. 

![demographic-columns](assets/demographic-columns.jpg)

Based on what our data dictionary tells us: this is a married college-educated female who was predicted to not default on this loan (got approved for the loan). Let’s try adjusting her education down and changing marital status to *single*. The assumption here is that in credit lending, single people without a college degree should not be denied credit or be predicted to default just because of this idea of being single and having an advanced education. 

*Note:* 

- Gender (1 = male; 2 = female)
 - Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)
- Marital status (1 = married; 2 = single; 3 = others) 

To change the **Education** and **Marital Status**, we will follow the same steps when we change *PAY_0* and *PAY_AMT2*.

After lowering the **Education** and setting the **Marital Status** to *Single (2)* and *rescoring* the table you should see something similar: 

![change-column-values](assets/change-column-values.jpg)

After changing the values, what do you discover?

As you will see, the prediction is not flipped, and this is good because a model in production should not have its prediction change because certain sensitive demographic variables have been modified. If it were the case that a prediction would change as a result of manipulating a specific demographic variable, this would tell us that bias has been introduced in our AI model that will result in legal consequences and penalties for your company or entity. As a result, if bias is present, developers of the AI model should use their technologies or common sense intuition to understand the ins of their model and fix the bias and as questions around; "does this seem ethical? Does this seem logical? These two questions are fundamental questions to ask yourself to provide fairness and avoid legal, ethical, and moral consequences. As well, these questions will guide you to decide on the definition of what is fair in your respective use case of the AI model.  

You might be asking yourself now why then the prediction was not flipped; well, recall the *Partial Dependence Plot* as discussed above the *Partial Dependence Plot* concluded that *PAY_0* was the variable that most determine whether someone will default or not. In particular, we also discuss that someone increases its probability of defaulting when *PAY_0 = 2*.

*Note:* 

This AI model that we develop at the start of task 5 made use of all columns and, therefore, took into consideration the demographic variables. In contrast, the first experiment we ran didn't use the demographic columns, so now the question is why we kept them? Isn't it illegal to make use of such demographic columns in production? And the answer to that question is yes, it's illegal. We kept them because we wanted to make use of sensitive columns and demonstrate how we would make use of DAI to check for sensitivity around certain variables if we were to use such demographic variables in our AI model in production. Note one should never make use of such columns in production. Furthermore,  if it were the case that we dropped the demographic columns at the start of task 5, we wouldn't have been able to see a potential bias in the model originating from the following columns: **AGE**, **SEX**, **EDUCATION**, and **MARRIAGE**.  Hence, a way to analyze for bias while dropping the demographic columns is by doing a residual analysis and looking at the constant global matrix. We would not cover this in this tutorial, but on the next one (COMING SOON), we will see how we can make use of a residual analysis and a constant global matrix to detect for unfair bias. 

A way we can flip the prediction for customer 3 is by changing *PAY_0 = 2*. If it's the case that the prediction changes, we will, at the same time, be confirming that being two months late on *PAY_0* will increase your chances of being predicted as *default*.  

![after-values-change](assets/after-values-change.jpg)

We see that adjusting *PAY_0 = 2* sends the customer very deep into the rejection zone, showing pretty clear how this one feature can dramatically impact the model for a single customer!

*Note:*

For this particular dataset, the one variable that, for the most part, determines whether someone will default or not is *PAY_0*. In these cases, developing an AI Model is not necessary given that by just looking at *PAY_0* one can predict in a manner of speaking, whether someone will default or not. 

### Conclusion

As mentioned at the beginning of this tutorial, the disparate impact analysis, and sensitivity analysis tool can help you understand your model's inner workings. Due to regulatory pressures and the aggressive adoption of AI in the enterprise, you are now more often required to be able to review and explain the ins of your model. It is worth noting that being able to explain a model after being built is just one component of responsible AI practices. 

## Next Steps

Check out the next tutorial: Analyzing a Criminal Risk Scorer with DAI MLI (COMING SOON), where you will learn more about:

 - Disparate Impact Analysis 
 - Sensitivity Analysis 
 - Confusion Matrix
 - Residual Analysis 
 - False Positive Rate
 - True Positive Rate



e demographic factors in how we treat and analyze a customer. Additionally, **Limit_Bal** has some adverse action considerations, and this variable is usually dictated internally, and therefore, we will also drop it. Click **Done**.