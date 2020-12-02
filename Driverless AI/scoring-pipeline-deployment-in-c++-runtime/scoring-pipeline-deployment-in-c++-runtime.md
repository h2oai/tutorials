# Scoring Pipeline Deployment in C++ Runtime

## Outline

- [Objective](#objective)
- [Prerequisites](#prerequisites)
- [Task 1: Set Up Environment](#task-1-set-up-environment)
- [Task 2: Concepts Around Scoring Pipeline in C++ Runtime](#task-2-concepts-around-scoring-pipeline-in-c++-runtime)
- [Task 3: Batch Scoring via Scoring Pipeline Execution](#task-3-batch-scoring-via-scoring-pipeline-execution)
- [Task 4: Challenge](#task-4-challenge)
- [Next Steps](#next-steps)
- [Appendix A: Glossary](#appendix-a-glossary)

## Objective

**Machine Learning Model Deployment** is the process of making your model available in production environments, so it can be used to make predictions for other software systems [1]. Before model deployment, **feature engineering** occurs in the form of preparing data that will later be used to train a model [2]. Driverless AI **Automatic Machine Learning (AutoML)** combines the best feature engineering and one or more **machine learning models** into a scoring pipeline [3][4]. The **scoring pipeline** is used to score or predict data when given new test data [5]. The scoring pipeline comes in two flavors. The first scoring pipeline is a **Model Object, Optimized(MOJO) Scoring Pipeline**, a standalone, low-latency model object designed to be easily embeddable in production environments. The second scoring pipeline is a Python Scoring Pipeline, which has a heavy footprint that is all Python and uses the latest libraries of Driverless AI to allow for executing custom scoring recipes[6].

By the end of this tutorial, you will predict the **cooling condition** for a **Hydraulic System Test Rig** by deploying an **embeddable MOJO Scoring Pipeline** into C++ Runtime using **Python** and **R**. 

### Resources

[1] H2O.ai Community AI Glossary: [Machine Learning Model Deployment](https://www.h2o.ai/community/glossary/machine-learning-model-deployment-productionization-productionizing-machine-learning-models)

[2] H2O.ai Community AI Glossary: [Feature Engineering](https://www.h2o.ai/community/glossary/feature-engineering-data-transformation)

[3] H2O.ai Community AI Glossary: [Automatic Machine Learning (AutoML)](https://www.h2o.ai/community/glossary/automatic-machine-learning-automl)

[4] H2O.ai Community AI Glossary: [Machine Learning Model](https://www.h2o.ai/community/glossary/machine-learning-model)

[5] H2O.ai Community AI Glossary: [Scoring Pipeline](https://www.h2o.ai/community/glossary/scoring-pipeline)

[6] H2O.ai Community AI Glossary: [Model Object, Optimized (MOJO) Scoring Pipeline](https://www.h2o.ai/community/glossary/model-object-optimized-mojo)

[7] [SAVERY - HYDRAULIC TEST RIGS AND BENCHES](https://www.savery.co.uk/systems/test-benches)

[8] [HYDROTECHNIK - Flow and Temperature Testing Components](https://www.hydrotechnik.co.uk/flow-and-temperature-hydraulic-test-bed)


## Prerequisites

- Skilled in Python and/or R Programming
- Driverless AI Environment
- Driverless AI License
    - The license is needed to use the **MOJO2 C++ Runtime Python Wrapper API** and **R Wrapper API** to execute the **MOJO Scoring Pipeline** to make predictions
    - If you don't have a license, you can obtain one through our [21 day trial license](https://www.h2o.ai/try-driverless-ai/) option. Through the [21 day trial license](https://www.h2o.ai/try-driverless-ai/) option, you will be able to obtain a temporary Driverless AI License Key necessary for this tutorial.
    - If you need to purchase a Driverless AI license, reach out to our sales team via the [**contact us form**](https://www.h2o.ai/company/contact/).
- Linux OS (x86 or IBM Power PC) or Mac OS X (10.9 or newer)
- Anaconda or Miniconda
- Basic knowledge of Driverless AI or completion of the following tutorials:
    - [Tutorial 1A: Automatic Machine Learning Introduction with Driverless AI Test Drive](https://training.h2o.ai/products/tutorial-1a-automatic-machine-learning-introduction-with-driverless-ai)
    - [Tutorial 4A: Scoring Pipeline Deployment Introduction](https://training.h2o.ai/products/tutorial-4a-scoring-pipeline-deployment-introduction#tab-product_tab_contents__12)
    - [Tutorial 4B: Scoring Pipeline Deployment Templates](https://training.h2o.ai/products/tutorial-4b-scoring-pipeline-deployment-templates)

## Task 1: Set Up Environment

For this tutorial, we will continue making use of the prebuilt experiment: **Model_deployment_HydraulicSystem.**  The Driverless AI  experiment is a classifier model that classifies whether the **cooling condition** of a **Hydraulic System Test Rig** is 3, 20, or 100. By looking at the **cooling condition,** we can predict whether the Hydraulic Cooler operates **close to total failure**, **reduced efficiency**, or **full efficiency**. 

| Hydraulic Cooling Condition | Description |
|:--:|:--:|
| 3 | operates at close to total failure |
| 20 | operates at reduced efficiency |
| 100 | operates at full efficiency |

The Hydraulic System Test Rig data for this tutorial comes from the **[UCI Machine Learning Repository: Condition Monitoring of Hydraulic Systems Data Set](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems#)**. The data set was experimentally obtained with a hydraulic test rig. This test rig consists of a primary working and a secondary cooling-filtration circuit connected via the oil tank [1], [2]. The system cyclically repeats constant load cycles (duration 60 seconds) and measures process values such as pressures, volume flows, and temperatures. The condition of four hydraulic components (cooler, valve, pump, and accumulator) is quantitatively varied. The data set contains raw process sensor data (i.e., without feature extraction), structured as matrices (tab-delimited) with the rows representing the cycles and the columns the data points within a cycle.


Hydraulic System Test Rigs are used to test Aircraft Equipment components, Automotive Applications, and more [1]. A Hydraulic Test Rig can test a range of flow rates that can achieve different pressures with the ability to heat and cool while simulating testing under different conditions [2]. Testing the pressure, the volume flow, and the temperature is possible by Hydraulic Test Rig sensors and a digital display. The display panel alerts the user when certain testing criteria are met while displaying either a green or red light [2]. Further, a filter blockage panel indicator is integrated into the panel to ensure the Hydraulic Test Rig's oil is maintained [2]. In the case of predicting cooling conditions for a Hydraulic System, when the cooling condition is low, our prediction will tell us that the cooling of the Hydraulic System is close to total failure, and we may need to look into replacing the cooling filtration solution soon. In our case, the shared project classifies the probability of the cooler condition is 3, 20, or 100.  

![cylinder-diagram-1](./assets/hydraulic-system-diagram.jpg)

**Figure 1:** Hydraulic System Cylinder Diagram

### Create Environment Directory Structure

```bash
# Create directory structure for DAI MOJO C++ Projects

# Create directory where the mojo-pipeline folder will be stored
mkdir $HOME/dai-mojo-cpp/

```

### Set Up Driverless AI MOJO Requirements

#### Download the MOJO Scoring Pipeline

1\. If you have not downloaded the MOJO Scoring Pipeline, consider the following steps: 

- Start a new Two-Hour Test Drive Session in Aquarium 
- In your Driverless AI instance, click on the Experiments section
- In the Experiments section, click on the following experiment: **Model_deployment_HydraulicSystem**
- On the **STATUS:COMPLETE** section on the experiment page, click **DOWNLOAD MOJO SCORING PIPELINE**
- In the **Java** tab, click **DOWNLOAD MOJO SCORING PIPELINE**

When finished, come back to this tutorial. 

2\. Move the **mojo.zip** file to the **dai-mojo-cpp** folder and then extract it:

```bash
cd $HOME/dai-mojo-cpp/
# Depending on your OS, sometimes the mojo.zip is unzipped automatically and instead of mojo.zip, write mojo-pipeline for the first command. If it's mojo-pipeline no need to execute the unzip command.  
mv $HOME/Downloads/mojo.zip .
unzip mojo.zip
```

#### Download the MOJO2 Python and R Runtime

We can download the **MOJO2 C++ Runtime Python Wrapper API and R Wrapper API** in Driverless AI. There are two places where we can download the MOJO2 Python and R runtime. 

1\. Similar to where we downloaded the **MOJO SCORING PIPELINE**, we will download the **Download the MOJO2 Py Runtime**.   Instead of clicking on the **Java** tab, click on the **Python** tab. Right after, click on the **Download the MOJO2 Py Runtime** hyperlink. Select the type of download you want, depending on your OS. 

![download-mojo2-py-runtime-1](assets/download-mojo2-py-runtime-1.jpg)

Similar for the **MOJO2 R runtime**, click the **R** tab, then **Download the MOJO2 R Runtime** hyperlink. Select the type of download you want, depending on your OS. 

![download-mojo2-r-runtime-2](assets/download-mojo2-r-runtime-2.jpg)

2\. The second place you can find these runtimes is under the **Resources** drop-down list.

![download-mojo2-py-r-runtime-3](assets/download-mojo2-py-r-runtime-3.jpg)

Click **MOJO2 Py Runtime** and/or **MOJO2 R Runtime** to download the runtime.

Now that you have downloaded the **MOJO2 C++ Runtime Python Wrapper API and R Wrapper API**, download and install Anaconda:

```bash
# Download Anaconda (Note: the command is for a Linux environment)
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

# Install Anaconda (Note: the command is for a Linux environment)
bash Anaconda3-2020.02-Linux-x86_64.sh

# To Download and Install Anaconda follow the steps on this link: https://docs.anaconda.com/anaconda/install/mac-os/
```

Move the **MOJO2 Py Runtime** file to the $HOME folder:

```bash
cd $HOME

# If you have a Mac, move the MOJO2 Py runtime for Mac OS X to the $HOME folder
mv $HOME/Downloads/daimojo-2.4.8-cp36-cp36m-macosx_10_7_x86_64.whl .
 
# If you have Linux, move the MOJO2 Py runtime for Linux x86 to the $HOME folder
mv $HOME/Downloads/ddaimojo-2.4.8-cp36-cp36m-linux_x86_64.whl .
 
# If you have a Linux PPC, move the MOJO2 Py runtime for Linux PPC to the $HOME folder
mv $HOME/Downloads/daimojo-2.4.8-cp36-cp36m-linux_ppc64le.whl .
```


Move the **MOJO2 R Runtime** file to $HOME folder:

```bash
cd $HOME
# If you have a Mac, move the MOJO2 R runtime for Mac OS X to the $HOME folder
mv $HOME/Downloads/daimojo_2.4.8_x86_64-darwin.tar.gz .
 
# If you have Linux, move the MOJO2 R runtime for Linux x86 to the $HOME folder
mv $HOME/Downloads/daimojo_2.4.8_x86_64-linux.tar.gz .
 
# If you have a Linux PPC, move the MOJO2 R runtime for Linux PPC to the $HOME folder
mv $HOME/Downloads/daimojo_2.4.8_ppc64le-linux.tar.gz .
```


### Install MOJO2 Python and R Runtime Dependencies

Create virtual environment and install Python and R packages in it

```bash
# Install Python 3.6.10
conda create -y -n model-deployment python=3.6
conda activate model-deployment
```

Install Python Packages

```bash
# Install Python Packages
# Install datable 0.10.1
pip install datatable
# Install pandas
pip install pandas
# Install scipy
pip install scipy
```

Depending on your OS, run one of the following commands to install the **MOJO2 Py Runtime**:

```bash
# Install the MOJO runtime on Mac OS X
pip install daimojo-2.4.8-cp36-cp36m-macosx_10_7_x86_64.whl

# Install the MOJO runtime on Linux PPC
pip install daimojo-2.4.8-cp36-cp36m-linux_ppc64le.whl

# Install the MOJO runtime on Linux x86
pip install daimojo-2.4.8-cp36-cp36m-linux_x86_64.whl
```

Install R packages

```bash
# Install R Packages
# Install R r-essentials 3.6.0
conda install -y -c r r-essentials

# Install R r-rcpp 1.0.3
conda install -y -c conda-forge r-rcpp=1.0.3

# Install R data.table
conda install -y -c r r-data.table
```

### Set Driverless AI License Key

Set the Driverless AI License Key as a temporary environment variable.

Note: If you don't have a license, you can obtain one through our [21-day trial license](https://www.h2o.ai/try-driverless-ai/) option. Through the [21-day trial license](https://www.h2o.ai/try-driverless-ai/) option, you will be able to obtain a temporary **Driverless AI License Key** necessary for this tutorial. 

```bash
# Set Driverless AI License Key
export DRIVERLESS_AI_LICENSE_KEY="{license-key}"
```

## Task 2: Concepts Around Scoring Pipeline in C++ Runtime  

### MOJO Scoring Pipeline Files

After downloading the MOJO scoring pipeline, the **mojo-pipeline** folder comes with many files needed to execute the MOJO scoring pipeline, including **pipeline.mojo** and **example.csv**. However, the **mojo-pipeline** folder does not come with the MOJO2 Py Runtime or MOJO2 R Runtime. These two MOJO2 APIs can be downloaded as separate assets from Driverless AI. The **pipeline.mojo** is the standalone scoring pipeline in MOJO format. This pipeline file contains the packaged feature engineering pipeline and the machine learning model. The **daimojo-2.4.8-cp36-cp36m-{OS: mac, linux. IBM Power}.whl** is the MOJO2 Python API. The **daimojo_2.4.8_{OS: mac, linux, IBM Power}.tar.gz** is the MOJO2 R API. The **example.csv** contains sample test data. 

### Embed the MOJO in the C++ Runtime via Python or R Wrappers

If you have gone through the earlier scoring pipeline deployment tutorials, you have seen how we deploy the MOJO Scoring Pipeline to a server or serverless instance. Some clients interact with the server to trigger it to execute the MOJO to make predictions. An alternative way to deploy the MOJO Scoring Pipeline is to embed directly into the C++ Runtime, where your application is running. The MOJO C++ Runtime comes with Python and R wrappers called MOJO2 Py Runtime and MOJO2 R Runtime.  Suppose you are building a Python or R application using an **Integrated Development Environment (IDE)** or a text editor. In that case, you can import the MOJO2 Python API or MOJO2 R API. Then use it to load the MOJO, put your test data into a MOJO frame, then perform predictions on the data to obtain results.

## Task 3: Batch Scoring via Scoring Pipeline Execution

We will be executing the MOJO scoring pipeline using the Python and R wrapper. We will be doing batch scoring on the Hydraulic System example CSV data to classify for the Hydraulic System cooling condition.

### Batch Scoring via Run R Wrapper Program

1\. Enter **R** to enter R's interactive terminal:

```bash
R
```

![batch-scoring-via-run-r-program-1](assets/batch-scoring-via-run-r-program-1.jpg)

2\. Now that we are in the R interactive terminal, we will install the MOJO2 R Runtime:


```bash
# Install the R MOJO runtime using one of the methods below

homePath <- Sys.getenv("HOME")

# Install the R MOJO runtime on PPC Linux
path <- paste(homePath, "/daimojo_2.4.8_ppc64le-linux.tar.gz", sep="")
install.packages(path, repos = NULL, type="source")

# Install the R MOJO runtime on x86 Linux
path <- paste(homePath, "/daimojo_2.4.8_x86_64-linux.tar.gz", sep="")
install.packages(path, repos = NULL, type="source")

#Install the R MOJO runtime on Mac OS X
path <- paste(homePath, "/daimojo_2.4.8_x86_64-darwin.tar.gz", sep="")
install.packages(path, repos = NULL, type="source")

```
**Note**: It might be the case that R will ask you to choose a download mirror. This is just the location where you want to download the package from. Pick one of the mirrors in the US, such as mirror 75 or the closest to your location. 

3\. Next, we will load the Driverless AI MOJO library and load the MOJO scoring pipeline:

```bash
# Load the MOJO
library(daimojo)
path <- paste(homePath, "/dai-mojo-cpp/mojo-pipeline/pipeline.mojo", sep="")
m <- load.mojo(path)
```

4\. We will then retrieve the creation time of the MOJO and the UUID of the experiment:

```bash
# retrieve the creation time of the MOJO
create.time(m)

# retrieve the UUID of the experiment
uuid(m)
```

5\. We will then set feature data types, names in the column class header, which will be used to initialize the R data table header and data types, and we will load the Hydraulic System example CSV data into the table.

```bash
# Load data and make predictions
col_class <- setNames(feature.types(m), feature.names(m))  # column names and types

library(data.table)
path <- paste(homePath, "/dai-mojo-cpp/mojo-pipeline/example.csv", sep="")
d <- fread(path, colClasses=col_class, header=TRUE, sep=",")
```

6\. Lastly, we will use our MOJO scoring pipeline to predict the Hydraulic System’s cooling condition for each row within the table:

```bash
predict(m, d)
```

![batch-scoring-via-run-r-program-2](assets/batch-scoring-via-run-r-program-2.jpg)

This classification output is the batch scoring done for our Hydraulic System cooling condition. You should receive classification probabilities for cool_cond_y.3, cool_cond_y.20, and cool_cond_y.100. The 3 means the Hydraulic cooler is close to operating at total failure, 20 means it is operating at reduced efficiency, and 100 means operating at full efficiency.

The results will give you a probability (a decimal value) for cool_cond_y.3, cool_cond_y.20, and cool_cond_y.100. After converting each decimal value to a percentage, note that the highest percentage per row will determine the type of cool_cond_y. 

7\. Quit the R interactive terminal:

```bash
quit()
```

R will ask you if you want to save the workspace image; feel free to save it if you wish.

Therefore that is how you execute the MOJO scoring pipeline to do batch scoring for the Hydraulic System cooling condition using the R wrapper in the C++ Runtime. 

### Batch Scoring through the Run Python Wrapper Program

1\. Enter **python** to enter Python's interactive terminal:

![batch-scoring-via-run-py-program-1](assets/batch-scoring-via-run-py-program-1.jpg)

2\. Let’s import the Driverless AI MOJO model package and load the MOJO scoring pipeline:

```bash
# import the daimojo model package
import os.path
import daimojo.model

homePath = os.path.expanduser("~")
# specify the location of the MOJO
m = daimojo.model(homePath + "/dai-mojo-cpp/mojo-pipeline/pipeline.mojo")
```

3\. We will then retrieve the creation time of the MOJO and the UUID of the experiment:

```bash
# retrieve the creation time of the MOJO
m.created_time

# retrieve the UUID of the experiment
m.uuid
```

4\. We can also retrieve a list of missing values, feature names, feature types, output names, and output types:

```bash
# retrieve a list of missing values
m.missing_values
# retrieve the feature names
m.feature_names
# retrieve the feature types
m.feature_types
# retrieve the output names
m.output_names
# retrieve the output types
m.output_types
```

5\. Now we will import the Python Datatable package, load the Hydraulic System example CSV data into the Datatable, set the table to ignore strings that equal to the missing values, and lastly, we will display the table:

```bash
# import the datatable module
import datatable as dt

# parse the example.csv file
pydt = dt.fread(homePath + "/dai-mojo-cpp/mojo-pipeline/example.csv", na_strings=m.missing_values, header=True, separator=',')

pydt
```

6\. We can also display the table column types:

```bash
# retrieve the column types
pydt.stypes
```

7\. We will use our MOJO scoring pipeline to predict the Hydraulic System’s cooling condition for each row within the table:

```bash
# make predictions on the example.csv file
res = m.predict(pydt)

# retrieve the predictions
res
```

![batch-scoring-via-run-py-program-2](assets/batch-scoring-via-run-py-program-2.jpg)

This classification output is the batch scoring done for our Hydraulic System cooling condition. You should receive classification probabilities for cool_cond_y.3, cool_cond_y.20, and cool_cond_y.100. The 3 means the Hydraulic cooler is close to operating at total failure, 20 means it is operating at reduced efficiency, and 100 means operating at full efficiency.

8\. There is some more data we can retrieve from our **res** predictions, which include the prediction column names and column types:


```bash
# retrieve the prediction column names
res.names

# retrieve the prediction column types
res.stypes
```

9\. We can also convert the datatable results to other data structures, such as pandas, NumPy, and list:

```bash
# need pandas
res.to_pandas()

# need numpy  
res.to_numpy()

# need list
res.to_list()
```

You just learned how to execute the MOJO scoring pipeline to do batch scoring for the Hydraulic System cooling condition using the Python wrapper in the C++ Runtime.

## Task 4: Challenge

### Execute Scoring Pipeline for a New Dataset

You could do something that helps you in your daily life or job. Maybe you could reproduce the steps we did above, but for a new experiment or dataset. In that case, you could either decide to do batch scoring, interactive scoring, or both. 

### Embed Scoring Pipeline into an Existing Program

Another challenge could be to use the existing MOJO scoring pipeline we executed. Instead of using the examples, we shared above, integrate the scoring pipeline into an existing Python or R program.

## Next Steps 

- [Tutorial 4C: Scoring Pipeline Deployment in Java Runtime](https://training.h2o.ai/products/tutorial-4c-scoring-pipeline-deployment-in-java-runtime#tab-product_tab_contents__9)
- [Tutorial 4E: Scoring Pipeline Deployment in Python Runtime](https://training.h2o.ai/products/tutorial-4e-scoring-pipeline-deployment-in-python-runtime#tab-product_tab_contents__9) 
- [Tutorial 4F: Scoring Pipeline Deployment to Apache NIFI](https://training.h2o.ai/products/tutorial-4f-scoring-pipeline-deployment-to-apache-nifi#tab-product_tab_contents__9)

## Appendix A: Glossary

Refer to the [H2O.ai Glossary](https://www.h2o.ai/community/top-links/ai-glossary-search?p=3) for relevant Model Deployment Terms