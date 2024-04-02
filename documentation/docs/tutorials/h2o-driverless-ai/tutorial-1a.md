---
title: Tutorial 1A
---

# Tutorial 1A: Automatic Machine Learning Introduction with Driverless AI

## Overview 

- This tutorial explores the Titanic dataset from the perspective of a passenger life insurance company while using and learning about H2O.ai's enterprise product, Driverless AI. We will explore possible risk factors derived from this dataset that could have been considered when selling passenger insurances. More specifically, we will create a predictive model to determine what factors contributed to a passenger surviving.

- Alongside the core analysis, you'll gain hands-on experience with Driverless AI. We'll cover essential functions like data loading, exploration of data details, generate Auto visualizations, experiment setup, explore feature engineering, and result interpretation through the Machine Learning Interpretability report.

- By the end, you'll have a good grasp of both Driverless AI's capabilities and the factors influencing passenger survival on the Titanic, all from the perspective of a life insurance company.


## Prerequisites

- Basic knowledge of Machine Learning and Statistics

- A Two-Hour Test Drive session: Test Drive is H2O.ai's Driverless AI on the AWS Cloud. No need to download software. Explore all the features and benefits of the H2O Automatic Learning Platform.
> - Need a Two-Hour Test Drive session? Follow the instructions [on this quick self-paced course](https://h2oai.github.io/tutorials/getting-started-with-driverless-ai-test-drive/#0) to get a Test Drive session started. After the Driverless AI Test Drive session starts, continue reading the remaining prerequisites of this self-paced course, then start Task 1: Product Tour.
> - Already have a Two-Hour Test Drive session? Proceed to [Task 1: Product Tour](#task-1-product-tour)

## Task 1: Product Tour

### Overview
H2O Driverless AI is an artificial intelligence(AI) platform for automatic machine learning. Driverless AI automates some of the most difficult data science and machine learning workflows such as feature engineering, model validation, model tuning, model selection, and model deployment. It aims to achieve highest predictive accuracy, comparable to expert data scientists, but in much shorter time thanks to end-to-end automation. Driverless AI also offers automatic visualizations and machine learning interpretability(MLI). Especially in regulated industries, model transparency and explanation are just as important as predictive performance. Modeling pipelines (feature engineering and models) are exported (in full fidelity, without approximations) both as Python modules and as pure Java standalone scoring artifacts.

### Why Driverless AI?
Over the last several years, machine learning has become an integral part of many organizations' decision-making process at various levels. With not enough data scientists to fill the increasing demand for data-driven business processes, H2O.ai offers Driverless AI, which automates several time consuming aspects of a typical data science workflow, including data visualization, feature engineering, predictive modeling, and model explanation.

H2O Driverless AI is a high-performance, GPU-enabled computing platform for automatic development and rapid deployment of state-of-the-art predictive analytics models. It reads tabular data from plain text sources, Hadoop, or S3 buckets and automates data visualization and building predictive models. Driverless AI targets business applications such as loss-given-default, probability of default, customer churn, campaign response, fraud detection, anti-money-laundering, demand forecasting, and predictive asset maintenance models. (Or in machine learning parlance: common regression, binomial classification, and multinomial classification problems).

### Tour

Welcome to the Driverless AI **Datasets** page (this will be the first thing you will see when you click your **Driverless AI URL**): ![Product Tour](product_tour.png)

On the **Datasets** page, the following options and features can be found. Now, we will quickly review them, but we will further explore these options and features before and after we launch an experiment in the upcoming tasks.

:::info note 
1.  In the **Projects** section, you can manage your datasets and experiments related to a specific business problem or use case.
2. In the **Datasets** section, you can view current datasets that have been added. Other dataset features include the option to get dataset details, visualize, split, predict, rename, download, and delete.
3. **Autoviz** shows graphs explaining the selected dataset to visualize. Note that the graphs on the visualization page can vary based on the information in your dataset. You can also view and download logs that Driverless AI generated during the visualization.
4. In the **Experiments** section, you can view completed experiments. Experiments can be revised or deleted.
5. In the **Diagnostics** section, you can diagnose a model and view model performance for multiple scorers based on the existing model and dataset.
6. In the **MLI (Machine Learning Interpretability)** section, you can interpret a model and view a list of already interpreted models.
7. The **Deployments** section allows you to access the MOJO and Python scoring pipelines for you to test or integrate into a final product. You can also deploy locally or in the cloud.

<details>
<summary>The <b>Resources</b> dropdown menu provides you with the following options to view or download:</summary>
<p>

- System Info
- Python Client
- R Client
- MOJO2 Java Runtime
- MOJO2 Py Runtime
- MOJO 2 R Runtime
- Tutorials
- Help
- About

</p>
</details>

<details>
<summary>The <b>User</b> option offers the following options:</summary>
<p>

- <b>Messages[ ]</b>: View upcoming events regardinging Driverless AI.
- <b>Logout</b>: Logs you out of your current session.

</p>
</details>

8. **<:** Takes you back to the previous page
9. **H2O.ai**: Takes you back to the H2OAI **Datasets** page.
10. **Driverless AI 1.X.X**: Version of Driverless AI


<details>
<summary>The <b>+ Add a Dataset(or Drag and Drop)</b> button provides the following options:</summary>
<p>

- <i>File System</i>: Search for datasets in our repository
- <i>Upload File</i>: Upload a dataset from your computer
- <i>Amazon S3</i>: Upload a dataset by using S3 buckets
- <i>Hadoop File System</i>: Upload a dataset from Hadoop
- <i>Data Recipe URL</i>: Upload a dataset with a customize data recipe reachable through a URL
- <i>Upload Data Recipe</i>: Upload a dataset with a customize data recipe located on your computer

</p>
</details>

11. **Type value to search for or date, e,g,** 15⁄09: Search for a given dataset by inputting its creation date.
:::

Before we load the dataset for our experiment, let us review some introductory concepts around Machine Learning.
#### Deeper Dive and Resources
- Learn more about H2O Driverless through the [H2O documentation](http://docs.h2o.ai/driverless-ai/latest-stable/docs/booklets/DriverlessAIBooklet.pdf).
- [Explore H2O Product Documentation](http://docs.h2o.ai/)
- [Learn more H2O Driverless by reviewing the FAQs](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/faq.html)


## Task 2: Automatic Machine Learning Concepts
### Artificial Intelligence and Machine Learning
The concepts found in this task are meant to provide a high-level overview of Machine Learning. At the end of this task, you can find links to resources that offer a more in-depth explanation of the concepts covered here.

Machine learning is a subset of Artificial intelligence where the focus is to create machines that can simulate human intelligence. One critical distinction between artificial intelligence and machine learning is that machine learning models "learn" from the data the models get exposed to. Arthur Samuel, a machine learning pioneer back in 1959, defined machine learning as a " field of study that gives computers the ability to learn without being explicitly programmed" [1]. A machine learning algorithm trains on a dataset to make predictions. These predictions are, at times, used to optimize a system or assist with decision-making.

### Machine Learning Training
Advances in technology have made it easier for data to be collected and made available. The available type of data will determine the kind of training that the machine learning model can undergo. There are two types of machine learning training, supervised and unsupervised learning. Supervised learning is when the dataset contains the output that you are trying to predict. For those cases where the predicting variable is not present, it's called unsupervised learning. Both types of training define the relationship between input and output variables.

In machine learning, the input variables are called features and the output variables labels. The labels, in this case, are what we are trying to predict. The goal is to take the inputs/variables/features and use them to come up with predictions on never-before-seen data. In linear regression, the features are the x-variables, and the labels are the y-variables. An example of a label could be the future price of avocados. In terms of feature examples regarding this self-paced course, in Task 3, we will see the following features when creating our survival prediction model: passenger class, sex, age, passenger fare, cabin number, etc.

A machine learning model defines the relationship between features and labels. Anyone can train a model by feeding it examples of particular instances of data. You can have two types of examples: labeled and unlabeled. Labeled examples are those where the X and Y values (features, labels) are known. Unlabeled examples are those where we know the X value, but we don't know the Y value[1]. Your dataset is similar to an example; the columns that will be used for training are the features; the rows are the instances of those features. The column that you want to predict is the label.

Supervised learning takes labeled examples and allows a model that is being trained to learn the relationship between features and labels. The trained model can then be used on unlabelled data to predict the missing Y value. The model can be tested with either labeled or unlabeled data. Note that H2O Driverless AI creates models with labeled examples.

### Data Preparation
A machine learning model is as good as the data that is used to train it. If you use garbage data to train your model, you will get a garbage model. With that said, before uploading a dataset into tools that will assist you with building your machine learning model, such as Driverless AI, ensure that the dataset has been cleaned and prepared for training. Transforming raw data into another format, which is more appropriate and valuable for analytics, is called data wrangling.

Data wrangling can include extractions, parsing, joining, standardizing, augmenting, cleansing, and consolidating until the missing data is fixed or removed. Data preparation includes the dataset being in the correct format for what you are trying to do; accordingly, duplicates are removed, missing data is fixed or removed, and finally, categorical values are transformed or encoded to a numerical type. Tools like [Python datatable](https://datatable.readthedocs.io/en/latest/?badge=latest), [Pandas](https://pandas.pydata.org/) and [R](https://www.r-project.org/) are great assets for data wrangling.

Data wrangling can be done in Driverless AI via a [data recipe](https://www.r-project.org/), the [JDBC connector](http://docs.h2o.ai/driverless-ai/1-8-lts/docs/userguide/connectors-nd/jdbc.html?highlight=jdbc) or through [live code](http://docs.h2o.ai/driverless-ai/1-8-lts/docs/userguide/datasets-describing.html?highlight=live%20code#modify-by-recipe) which will create a new dataset by modifying the existing one.

### Data Transformation/Feature Engineering
Data transformation or feature engineering is the process of creating new features from the existing ones. Proper data transformations on a dataset can include scaling, decomposition, and aggregation [2]. Some data transformations include looking at all the features and identifying which features can be combined to make new ones that will be more useful to the model's performance. For categorical features, the recommendation is for classes that have few observations to be grouped to reduce the likelihood of the model overfitting. Categorical features may be converted to numerical representations since many algorithms cannot handle categorical features directly. Besides, data transformation removes features that are not used or are redundant[3]. These are only a few suggestions when approaching feature engineering. Feature engineering is very time-consuming due to its repetitive nature; it can also be costly. After successfully or having a notion of well-done data transformation, the next step in creating a model is selecting an algorithm.

### Algorithm Selection
"Machine learning algorithms are described as learning a target function (f) that best maps input variables (x) to an output variable(y): Y= f(x)" [4]. In supervised learning, there are many algorithms to select from for training. The type of algorithm(s) will depend on the size of your data set, structure, and the type of problem you are trying to solve. Through trial and error, the best performing algorithms can be found for your dataset. Some of those algorithms include linear regression, regression trees, random forests, Naive Bayes, and boosting, to name a few [5].

### Model Training
#### Datasets
When training a machine learning model, one good practice is to split up your dataset into subsets: training, validation, and testing sets. A good ratio for the entire dataset is 70-15-15, 70% of the whole dataset for training, 15% for validation, and the remaining 15% for testing. The **training set** is the data used to train the model, and it needs to be big enough to get significant results from it. The **validation set** is the data held back from the training and will be used to evaluate and adjust the trained model's hyperparameters and, hence, adjust the performance. Finally, the **test set** is data that has also been held back and will be used to confirm the final model's results [1].

:::info note 
The validation dataset is used for tuning the modeling pipeline. If provided, the entire training data will be used for training, and validation of the modeling pipeline is performed with only this validation dataset. When you do not include a validation dataset, Driverless AI will do K-fold cross-validation for I.I.D. (identically and independently distributed) experiments and multiple rolling window validation splits for time series experiments. For this reason, it is not generally recommended to include a validation dataset as you are then validating on only a single dataset. Please note that time series experiments cannot be used with a validation dataset: including a validation dataset will disable the ability to select a time column and vice versa.
:::

This dataset must have the same number of columns (and column types) as the training dataset. Also, note that if provided, the validation set is not sampled down, so it can lead to large memory usage, even if accuracy=1 (which reduces the train size). In a moment, we will learn more about accuracy when preparing an experiment.[10] ![Dataset](dataset.jpg)

Another part of model training is fitting and tuning the models. For fitting and tuning, hyperparameters need to be tuned, and cross-validation needs to take place using only the training data. Various hyperparameter values will need to be tested. "A hyperparameter is a parameter that is set before the learning process begins. These parameters are tunable and can directly affect how well a model trains. [In Machine Learning, a hyperparameter is the learning rate]" [7]. In other words, the hyperparameter value is used to determine the rate at which the model learns.

With cross-validation, the whole dataset is utilized, and each model is trained on a different subset of the training data [8]. Additionally, a cross-validation loop will be set to calculate the cross-validation score for each set of hyperparameters for each algorithm. Based on the cross-validation score and hyperparameter values, you can select the model for each algorithm that has been tuned with training data and tested with your test set. Remember that "a model in [Machine Learning(ML)] is the output of an ML algorithm run on data. It represents what was learned by a machine learning algorithm." [9]

### What are the challenges in AI Model Development?
One of the significant challenges in developing a single production-ready model is that it can take weeks or months to build it. Developing a model involves feature engineering, model building, and model deployment. All tasks are very repetitive, time-consuming, require advanced knowledge of feature generation, algorithms, parameters, and model deployment. Finally, there needs to be in-depth knowledge and confidence in how the model was generated to justify how it made its decisions.

### What is Automated Machine Learning, and why is it important?
AutoML or Automated Machine Learning is the process of automating algorithm selection, feature generation, hyperparameter tuning, iterative modeling, and model assessment. AutoML tools such as H2O Driverless AI makes it easy to train and evaluate machine learning models. Automating the repetitive tasks around Machine Learning development allows individuals to focus on the data and the business problems they are trying to solve.

With this task in mind, let's explore and load the data that we will be using when predicting whether a passenger would've survived the titanic accident.

### References
- [1] [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data)
- [2] [About Train, Validation and Test Sets in Machine Learning](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)
- [3] [Data Science Primer - Data Cleaning](https://elitedatascience.com/data-cleaning)
- [4] [Feature Engineering](https://elitedatascience.com/feature-engineering)
- [5] [Towards Data Science- Supervised vs Unsupervised Learning](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d)
- [6] [Selecting the best Machine Learning Algorithm for your regression problem](https://towardsdatascience.com/selecting-the-best-machine-learning-algorithm-for-your-regression-problem-20c330bad4ef)
- [7] [Deep AI - What is a hyperparameter?](https://deepai.org/machine-learning-glossary-and-terms/hyperparameter)
- [8] [H2O.ai's Driverless AI - Internal Validation Technique](http://docs.h2o.ai/driverless-ai/1-8-lts/docs/userguide/internal-validation.html?highlight=cross%20validation)
- [10] [Validation Dataset](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/experiment-settings.html#validation-dataset)

### Deeper Dive and Resources
- [Explore the replays from H2O World Around the World](https://www.h2o.ai/h2oworldnewyork/)
- [Explore the Webinar Replays](https://www.brighttalk.com/search/?q=driverless+ai)
- [Explore the various H2O Driverless AI playlists on YouTube](https://www.youtube.com/user/0xdata/playlists)
## Summary 

## Next 
