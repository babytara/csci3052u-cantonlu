**Discrepancy Analysis**  
Majority class baselines were established for LD (Language Detection), SA (sentiment analysis) and natural language inference (NLI). The character level TF-IDF with logistic regression used for the three tasks significantly outperforms the majority class baselines. This can be seen in the majority class macro f1 scores of 0.27, 0.17, and 0.33 for LD, SA and NLI respectively. This is to be expected as the majority class examples occupy a major percentage of the set. The Tf-IDF models, which understand relationships between textual features, perform much better across class types, with macro f1 scores of 0.61, 0.74 and 0.7 respectively for LD, SA and NLI. 

Implementation differed from CantoNLU in terms of model architecture as well as evaluation metrics. N-gram range and C values were hardcoded and predefined, which differs from CantoNLU which used fine tuned pretrained BERT models, such as bert-base-chinese and bert-base-cantonese for specific tasks using task specific hyperparameters with differing batch sizes over the course of a number of epochs. Accuracy and macro f1 scores were provided for the LD and SA tasks while evaluation metrics provided by CantoNLU (Table 3, p. 7\) were exclusively f1 scores. Included accuracy metrics demonstrate how an overall accuracy can misrepresent how a model actually performs across classes. 

Compared to fine tuned BERT models, Char TF-IDF with logistic regression is more sensitive to noise 

Stratified sampling was used to utilize a smaller subset of examples with preserved label proportions, with the test set being used to obtain the evaluation metrics (baselines). CantoNLU differs by using its own benchmark dataset splits along with its fine turned BERT models. 

**Next steps** 

- Error analysis   
- Find limitations   
- Begin utilizing commercial LLMs on stratified samples  
- Obtain baselines for remaining tasks 

**Stop**

- Some tasks require prerequisite assigned tasks to be completed beforehand

**Continue** 

- Tuning procedure/pipeline  
- Comparing discrepancies between tasks   
- Analysing evaluation metrics between tasks 