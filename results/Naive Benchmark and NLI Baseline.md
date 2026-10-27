**Naive Benchmark and NLI Baseline**  
The naive benchmark predicts the most common training label for each test example in LD, sentiment, and NLI. Character n-gram TF-IDF features were used to build a logistic regression for NLI. For the NLI TF-IDF baseline, each example has a premise and a hypothesis, which receive a separate vectorizer that fits only on the training split, and their feature matrices are combined with logistic regression before classification. Given the large training set, the NLI classifier uses a stratified random sample of 100,000 pairs for efficient tuning, while validation and test sets remain the same. The best settings, as well as the character n-gram ranges, and C values were tested, where the best one was then selected using validation macro-F1. On the test set, the selected model uses n-grams \[1, 3\], and C \= 1.0, achieving a validation macro-F1 of approximately 0.6945.

NLI TF-IDF settings:

* Character n-gram ranges: (1,2), (1,3) and (1,4)  
* Logistic regression C: 0.1, 1.0, and 10.0  
* Maximum features: 200,000 for each vectorizer  
* Minimum document frequency (min\_df): 2 to ignore n-grams that appear in only one text and reduce the impact of frequent n-grams  
* Logistic regression maximum iterations: 2000  
* Chosen model metric: validation macro-F1  
* sublinear\_tf: to reduce the significance of repeated words  
* random seed: 42

**Results**

| task | model | split | n | accuracy | macro\_f1 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| LD | majority\_class | test | 2341 | 0.6608 | 0.2653 |
| sentiment | majority\_class | test | 999 | 0.3333 | 0.1667 |
| NLI | majority\_class | test | 1302 | 0.5 | 0.3333 |
| NLI | char\_tfidf\_logistic\_regression | test | 13202 | 0.7071 | 0.7070 |

The NLI TF-IDF achieves 20.71% more accuracy compared to the majority baseline. The LD majority baseline has an accuracy of 66.08%, but a macro-F1 of 0.2653, showing that accuracy isn't a sufficient metric for measuring the performance of the baseline on its own.  
