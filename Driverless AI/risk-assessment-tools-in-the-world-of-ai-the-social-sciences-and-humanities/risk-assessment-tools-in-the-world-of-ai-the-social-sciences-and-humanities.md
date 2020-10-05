# Tutorial 1E: Risk Assessment Tools in the World of AI, the Social Sciences, and Humanities

## Outline

- [Objective](#objective)
- [Prerequisites](#prerequisites)
- [Task 1: Create a New Criminal Risk Scorer in DAI](#task-1-create-a-new-criminal-risk-scorer-in-dai)
- [Task 2: Explore Criminal Risk Scorers in America and The Compass Dataset](#task-2-explore-criminal-risk-scorers-in-america-and-the-compass-dataset)
- [Task 3: Understand Task 1](#task-3-understand-task-1)
- [Task 4: A Global Versus Local Behavior Analysis Concept](#task-4-a-global-versus-local-behavior-analysis-concept)
- [Task 5: Global Behavior Analysis](#task-5-global-behavior-analysis)
- [Task 6: What Biases Exist in My Model? What Levels of Disparity Exist Between Population Groups?](#task-6-what-biases-exist-in-my-model?-what-levels-of-disparity-exist-between-population-groups?)
- [Task 7: Local Behavior Analysis](#task-7-local-behavior-analysis)
- [Task 8: The Reciprocal Impact that Should Exist Between AI, the Social Sciences, and Humanities](#Task-8-the-reciprocal-impact-that-should-exist-between-ai-the-social-sciences-and-humanities)
- [Next Steps](#next-steps)



## Objective

As of now, artificial intelligence is being integrated into our daily lives and at different levels of our society. For many, artificial intelligence is the key to a more prosperous future, and for others, artificial intelligence is a source of wrongness, if not understood completely. In recent years, many reputable news sources have pointed out that artificial intelligence has become the reason for many discriminatory actions. In particular, ProPublica, a nonprofit newsroom, concluded in 2016 that automated criminal risk assessment algorithms hold racial inequality. Throughout the United States, an array of distinct automated criminal risk assessment algorithms have been built. Sadly, as often happens with risk assessment tools, many are adopted without rigorous testing on whether it works. In particular, most don't evaluate existing racial disparities and parity in the AI model. That is why, after constructing a criminal risk assessment model, we need to check that the model is fair and that it does not hold any discrimination. Such a post-analysis can prevent the AI model from committing unwanted discriminatory actions.

With the above in mind, in this tutorial, and with the help of DAI, we will build a criminal risk assessment model, but in this case, rather than predicting someone's risk level, we will predict whether someone will be arrested within two years since a given arrest. Right after, we will analyze the AI model, and we will check for fairness, disparity, and parity. To debug and diagnose unwanted bias in our prediction system model, we will conduct a global versus local analysis in which global and local model behavior are compared. We will also conduct a disparate impact analysis to answer the following two questions: what biases exist in my model? And what levels of disparity exist between population groups? Immediately later, we will have an interdisciplinary conversation around risk assessment tools in the world of AI, the social sciences, and humanities.Singularly, we will discuss several societal issues that can arise if we implement our AI recidivism prediction system or someone else's risk assessment tool. In hopes of finding a solution to the issues that can arise, we will explore how the social sciences and humanities can solve these societal issues. Simultaneously, we will discuss the reciprocal impact that should exist between AI, the social sciences, and humanities (such as in philosophy) and how that reciprocal impact can lead to a more acceptable and fair integration of AI into our daily lives and the judicial system. 

All things considered, it is the authors' hope that this tutorial inspires the standard to consult the social sciences and humanities when creating AI models that can have a tremendous impact on multiple areas of our diverse society. It is also the hope to alert social scientists and members of the humanities to extend their social activism to AI model creation, which will have a considerable impact on their fields of study. In essence, this joint work between AI, the humanities, and the social sciences can lead us to a more equitable and fair integration of AI into our daily lives and society. 

## Prerequisites

You will need the following to be able to do this tutorial:

- Basic knowledge of Confusion Matrices 
- Basic knowledge of Driverless AI or doing the [Automatic Machine Learning Introduction with Driverless AI Test Drive](https://training.h2o.ai/products/tutorial-1a-automatic-machine-learning-introduction-with-driverless-ai) 
- Completion of the [Machine Learning Interpretability Tutorial](https://training.h2o.ai/products/tutorial-1c-machine-learning-interpretability-tutorial)
- Completion of the Disparate Impact Analysis Tutorial 
- Review the following two articles on ProPublica's investigations: 
  - [Machine Bias: There’s software used across the country to predict future criminals. And it’s biased against blacks.](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
  - [How We Analyzed the COMPAS Recidivism Algorithm](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)
- A **Two-Hour Test Drive session**: Test Drive is [H2O.ai's](https://www.h2o.ai) Driverless AI on the AWS Cloud. No need to download software. Explore all the features and benefits of the H2O Automatic Learning Platform.
  - Need a **Two-Hour Test Drive** session?Follow the instructions on this quick tutorial to get a Test Drive session started.

**Note: Aquarium’s Driverless AI Test Drive lab has a license key built-in, so you don’t need to request one to use it. Each Driverless AI Test Drive instance will be available to you for two hours, after which it will terminate. No work will be saved. If you need more time to further explore Driverless AI, you can always launch another Test Drive instance or reach out to our sales team via the contact us form.**

## Task 1: Create a New Criminal Risk Scorer in DAI

## Task 2: Explore Criminal Risk Scorers in America and The Compass Dataset

## Task 8: The Reciprocal Impact that Should Exist Between AI, the Social Sciences, and Humanities

Why the social sciences and the humanities? Because the social sciences and humanities can best solve the problems, AI is creating in our society. Plus, AI models will and are already impacting their several subfields of studies. To support this argument, consider the argue below: 

At this point, many will wonder how we can solve these issues presented at the global and local levels. A solution will be to have the humanities, and social sciences integrated and extend their work to the process of AI model creation. Also, have the AI world consult the humanities and social sciences when we could generate models that profoundly impact people's lives. To understand the reciprocal impact that should exist between AI, the social sciences, and humanities, we will explore five problems that can arise if we were to use our model or similar models made by others. Simultaneously, we will discuss how that reciprocal impact can lead to a more acceptable and fair integration of AI into our daily lives and the judicial system.


**Problem 1: Confrontation Clause**

To start our analysis, consider the opening statement Rebecca Wexler wrote in her New York Times article title When a Computer Program Keeps You in Jail:

“The criminal justice system is becoming automated. At every stage — from policing and investigations to bail, evidence, sentencing and parole — computer systems play a role. Artificial intelligence deploys cops on the beat. Audio sensors generate gunshot alerts. Forensic analysts use probabilistic software programs to evaluate fingerprints, faces and DNA. Risk-assessment instruments help to determine who is incarcerated and for how long.

Technological advancement is, in theory, a welcome development. But in practice, aspects of automation are making the justice system less fair for criminal defendants.

The root of the problem is that automated criminal justice technologies are largely privately owned and sold for profit. The developers tend to view their technologies as trade secrets. As a result, they often refuse to disclose details about how their tools work, even to criminal defendants and their attorneys, even under a protective order, even in the controlled context of a criminal proceeding or parole hearing.

Take the case of Glenn Rodríguez. An inmate at the Eastern Correctional Facility in upstate New York, Mr. Rodríguez was denied parole last year despite having a nearly perfect record of rehabilitation. The reason? A high score from a computer system called Compas. The company that makes Compas considers the weighting of inputs to be proprietary information.” As a result, Mr. Rodríguez wasn’t able to cross-examine the score (evidence) used against his parole denial. 

The above is a clear example of how AI is deciding whether someone will receive parole while not allowing a defendant to cross-examine the evidence. For legal practitioners and other social scientists, this will be in clear violation of the confrontation clause found in the constitution. The confrontation clause is as follows: 

“The Confrontation Clause of the Sixth Amendment to the United States Constitution provides that "in all criminal prosecutions, the accused shall enjoy the right…to be confronted with the witnesses [evidence] against him." Generally, the right is to have a face-to-face confrontation with witnesses who are offering testimonial evidence against the accused in the form of cross-examination during a trial. The Fourteenth Amendment makes the right to confrontation applicable to the states and not just the federal government.[1] The right only applies to criminal prosecutions, not civil cases or other proceedings.”

As of now, it isn't very easy to explain an AI model's predictions. And for the most part, the explanations of these AI models cannot be disclosed because they are protected by law. A lack of explanation of how predictions are derived will prevent the recidivism AI model's use if the predictions can't be explained; cross-examination will not be possible. This lack of explainability hasn't stopped courtrooms from using AI models to determine who is incarcerated and for how long. 
To address this problem, AI makers and law practitioners should discuss and work together when building recidivism models. The collaboration will allow for an understanding of why such models will be violating the law and how we can perhaps work around the problem of a lack of explainability. This joint work can also lead to the often practice to consult law practitioners before using models that can have substantial legal consequences. In our case, consulting legal practitioners can allow for a deeper understanding of how perhaps our model can affect the confrontations clause or other parts of the law. 

**Problem 2: Fair and unfair models will still be wrong in the legal field**

Even if we were to build recidivism models that can accurately predict whether someone will be arrested within two years since the last arrest, that will still not exclude the model from being labeled as wrong. Image for one moment that you are in court, and you are given more time behind bars because you have been predicted to be arrested again by this perfect AI model. In a way, you will be punished by something that you haven't committed. According to the law, you can't be punished behind bars for something you haven't done. Will it be wrong to keep someone in jail because you know that person will soon commit another crime? Will it be wrong for you to let that person go because the offense hasn't been committed, but will be quickly committed in the near future? Hence, the argument that fair and unfair models will still be wrong in the legal field. The question now is whether this will be wrong or not. In a way, this is a moral problem that I believe the humanities and social sciences can answer best. If it's the case that members of the social sciences and humanities conclude that it will be wrong, that will impact the AI world in a way that it will make their AI model creations unnecessary. Concerning us, this will affect the possible implementation of our built AI model.

**Problem 3: Unreachable perfect models**

In the world of AI, constant work is being done to achieve perfect AI models. Still, that work many have agreed is useless, given that we will never be able to attain perfect AI models given the complexity of AI model creation. However, we have models that can get pretty close to perfection. The only problem with these models will be that they will hold a certain level of wrongness and unfairness. For many, AI models predicting recidivism and with a small amount of unfairness will be acceptable, given that it will generate a lot of goodness. The question here will be if this argument will justify the existence of a certain level of unfairness in our model. In a way, social scientists can tell us the goods and bad this can generate in society and perhaps have the philosophy field tell us if it will be morally acceptable. Sadly, these interdisciplinarity conversations are often not occurring. Consequently, we lack a moral decision and consciousness on whether to leverage AI's power while allowing for a certain amount of imperfection/unfairness in our model. 

**Problem 4: Defining Fairness**

The AI world lacks a concrete moral definition of what will be considered an excellent fair model. The author's option is that by studying the several existing moral philosophies, we will clearly define fairness in the AI world. That is why more interdisciplinary conversations around AI and the social sciences and humanities are necessary. If we want to integrate AI into several aspects of our daily lives, we need to make sure that such AI will not create or amplify social problems. That is why we need a concrete definition of fairness to force every AI model to abide to the fairness criteria.   

**Problem 5: Prior Criminal Counts**

As discovered, it is clear that someone's prior criminal count can't be a clear reflection of true recidivism. Like prior counts, we also notice that age is another factor that shouldn't be considered when predicting recidivism. Correspondingly, we need to replace these traditionally used recidivism factors and explore for new unbiased features. Thus, we need joint work between AI, the social sciences, and the humanities if we hope to predict multiple recidivism types truly. 

**Final Analysis**

As a final analysis, the AI world is growing and changing our society. We need the social sciences and humanities to solve the problems AI is creating during its creation. This joint work can genuinely be the key to more acceptance of AI into our daily lives while leveraging AI's power. Most of the errors produced by AI models are impacting the most marginalized groups. These errors are causing a dislike towards AI models and are making the integration of AI into our society more difficult. This dislike needs to change, and the change starts when we work together to understand the several impacts these models can have on the most marginalized groups. Whether people are already integrating the social sciences and humanities to AI, we need to make sure that such integration is speedup because we currently have the wrong following titles in several news sources across the country: 

- When AI in healthcare goes wrong, who is responsible?
- Rise of the racist robots – how AI is learning all our worst impulses
- Racist Data? Human Bias is Infecting AI Development
- Google apologizes after its Vision AI produced racist results
- 600,000 Images Removed from AI Database After Art Project Exposes Racist Bias

## Next Steps

Check out the next tutorial: Time Series Tutorial - Retail Sales Forecasting where you will learn more about:

- Time-series:
- Time-series concepts
- Forecasting
- Experiment settings
- Experiment results summary
- Model interpretability
- Analysis
