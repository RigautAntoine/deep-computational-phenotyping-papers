# deep-computational-phenotyping-papers
An attempted survey of the current state of Deep Computational Phenotyping, i.e. the use of deep learning techniques for extracting insights from large unstructured Electronic Health Records. 

## Quick intro to Deep Computational Phenotyping

The goal of *Computational Phenotyping* is to discover descriptors of patient's health states, which can in turn be used for multiple applications, ranging from future risk identification to subject selection in clinical trials or treatment recommendation. Researchers have identified the opportunities presented by the proliferation of Electronic Health Records systems (the literature also uses the term Electronic Medical Records and the respective acronyms - EHRs/EMRs). This huge unstructured digitized source of insights typically takes the form of diagnosis codes, medication orders, operations, and medical narratives, and even include at times medical imagery and biomedical literature. 

Deep Computational Phenotyping takes advantages of the recent advances in neural network modeling and software to make sense of EMRs data. 

This page attempts to make a survey of this community and is intended for new students of the field. 


## The Challenges

I subdivide the challenges into two domains: (1) the unique challenges offered by the processing of EMR *data*; (2) the challenges of doing *inference* in a medical setting.

### The Data Issue 

- **Learning without Internet-sized big data**. Deep learning methods have prospered primarily in the natural language processing, computer vision, and speech recognition communities, all of which can rely on the availability of millions, hundreds of millions, of training examples. EMRs datasets have not grown to that scale yet, which makes deep learning models prone to overfitting. Many techniques have been proposed, from the careful use of regularization techniques (dropout, Bayesian priors), to transfer learning, to unsupervised weight initialization.

- **Dealing with class imbalance**. An extension of the prior challenge, but which takes a special importance in health care, where rare conditions affecting as low as 1% of our population of interest are the norm rather than the exception. Machine learning traditionally does poorly in tasks that face a class imbalance and the problem is further compounded in deep learning, which requires an even larger number of training examples to converge properly.

- **Noisy data**. EMRs data is fundamentally noisy, sparse, incomplete, high-dimensional, if not systematically biased. To illustrate with an example, in many cases a diagnosis code is recorded in the database to signal that laboratory tests are carried out to confirm or disprove the corresponding diagnosis, and is in no way an indication that the patient actually has the condition. Medical records are a mixture of confounding interactions between disease progression and intervention processes. Furthermore, medical conditions are recorded as high-dimensional sets of codes, numbering in the hundreds of thousands, which makes them difficult to process. When such data is used as the dependent variable, this translates to a *weak label* situation, given the problem of misdiagnosis, false positives, and undetected asymptomatic conditions in health care. 

### The Inference Issue

The availability of huge datasets is not the sole reason why deep learning has prospered in the NLP, computer vision and speech processing communities. I would also contend that these fields are held to far less restrictive standards than medical researchers, with regards to model interpretation, estimates of uncertainty, and causal analysis. All of these aspects continue to challenge the ability of deep learning models to this day and are areas of active research.

- **Measuring uncertainty in models**. Given a model and a prediction task, how certain is the model about the point estimate for that prediction? How stable are the model's parameters? Models in health care must come under scrutiny for robustness and prediction uncertainty. Such questions are still difficult to answer in deep neural networks, but active research is conducted on borrowing precepts from Bayesian inference, which is typically good at quantifying uncertainty.

- **Causal Analysis**. Risk factors used in predictive modeling for health care are rigorously selected and tested. How do we measure the causal impact of the learned features in a neural network? 

- **Opening the black-box**. Neural networks are still difficult to interpret. The research in determining *why* neural networks perform well and *what* they actually learn is still sparse. Creating and leveraging new methods of interpreation, visual analytics, and model diagnostics in a deep learning setting is an active area of research. 

- **Leveraging prior information**. Deep learning techniques were developed with the idea of *letting the data speak* with minimal input from experts. However, this approach is wasteful in the medical setting, which has accumulated decades of expert knowledge and where we cannot rely on the huge datasets that prevail in other fields. Again, this is in contrast to Bayesian inference, which takes the sum of prior knowledge and data evidence. Thus, we need to construct a framework for including all sorts of prior information, from Bayesian priors informed by experts, to hierarchical relations between outputs/inputs, to the medical ontologies built over the last half-century, to leveraging information from medical journals.  


## The Applications

Fundamentally, the goal of Computational Phenotyping is to derive a precise, informed and dense representation of a patient's health state. 

Such a descriptor would have countless applications:
- Automated diagnosis
- Predicting length of stay
- Cohort selection and recruitment for clinical trials
- Future risk detection
- Survival analysis and development of risk scores
- Anomaly detection (eg. vitals monitoring)
- Recommendation of treatment
- Disease progression modeling
- Unwanted readmission prediction
- Artifical assistants for physicians
- Parsing and mining of biomedical literature and medical notes and narratives.

To name only some of the obvious ones.


## The papers

### Core papers
Disclaimer: these are, in my personal opinion, the papers that addressed the challenges and applications in this field most directly, to the best of my current knowledge of the field. This list is intended for new arrivants in the field and is by no means exhaustive or finished. Papers were sorted alphabetically.

- [Causal Phenotype Discovery via Deep Networks](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4765623/)
- [Computational Phenotype Discovery Using Unsupervised Feature Learning over Noisy, Sparse, and Irregular Clinical Data](http://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0066341)
- [Directly Modeling Missing Data in Sequences with RNNs: Improved Classification of Clinical Time Series](https://arxiv.org/abs/1606.04130)
- [Doctor AI: Predicting Clinical Events via Recurrent Neural Networks](http://arxiv.org/abs/1511.05942)
- [Deep Computational Phenotyping](http://dl.acm.org/citation.cfm?id=2783365)
- [DeepCare: A Deep Dynamic Memory Model for Predictive Medicine](http://arxiv.org/abs/1602.00357)
- [Deep Patient: An Unsupervised Representation to Predict the Future of Patients from the Electronic Health Records](http://www.nature.com/articles/srep26094)
- [Learning to Diagnose with LSTM Recurrent Neural Networks](https://arxiv.org/abs/1511.03677)
- [Learning Robust Features using Deep Learning for Automatic Seizure Detection](https://arxiv.org/abs/1608.00220)
- [Learning Representations from EEG with Deep Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1511.06448)
- [PD Disease State Assessment in Naturalistic Environments Using Deep Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9930)
- [Unsupervised Learning of Disease Progression Models](http://dl.acm.org/citation.cfm?id=2623754)


### Other papers of interest (to read, assess, and complete)

- [Multi-task Learning with Weak Class Labels: Leveraging iEEG to Detect Cortical Lesions in Cryptogenic Epilepsy](https://arxiv.org/abs/1608.00148)
- [Deep Survival Analysis](https://arxiv.org/abs/1608.02158v1)
- Multi-layer Representation Learning for Medical Concepts
- [Deep Survival: A Deep Cox Proportional Hazards Network](https://arxiv.org/abs/1606.00931)
- [Rubik: Knowledge Guided Tensor Factorization and Completion for Health Data Analytics](http://dl.acm.org/citation.cfm?id=2783395&CFID=827394832&CFTOKEN=20057722)
- Towards Heterogeneous Temporal Clinical Event Pattern Discovery: A Convolutional Approach
- [Transferring Knowledge from Text to Predict Disease Onset](https://arxiv.org/abs/1608.02071)
- [Unsupervised Pattern Discovery in Electronic Health Care Data Using Probabilistic Clustering Models](http://dl.acm.org/citation.cfm?id=2110408)

## Datasets

Most common public datasets used to test methods in the deep computational phenotyping community:
- [MIMIC-II](https://physionet.org/mimic2/)
- [MIMIC-III](https://physionet.org/physiobank/database/mimic3cdb/): deidentified clinical care data collected at Beth Israel Deaconess Medical Center from 2001 to 2012. It contains over 58,000 hospital admission records of 38,645 adults and 7,875 neonates.
- Physionet challenge 2012: collection of multivariate clinical time series from 8000 ICU records. Each record is a multivariate time series of roughly 48 hours and contains 33 variables such as Albumin, heart-rate, glucose etc.
The entire [Physionet](https://physionet.org/) database 


## Future work

- Do a break-down by paper by theme, the challenge tackled and the solution contributed, perhaps in a table format.
- List of common freely available datasets with links used in the field and benchmarks to-date.
- List of useful workshop, conferences
- Do a full review of each paper
- There are by my estimation 40-50 papers to know in this field. I will enlarge, consolidate and sort this list.



