# Bag of Words Representation
## Naive Bayes Classifier

###Simple Tokenizer
| Alpha  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1  | 0.81  | 0.82312 | 0.916083916 |
| 5  | 0.83  | 0.83088 | 0.919080919 |
| 10  | 0.835  | 0.83392 | 0.919080919 |
| 15  | 0.8475  | 0.83532 | 0.918081918 |
| 20  | 0.8475  | 0.83532 | 0.914585415 |
| 25  | 0.85  | 0.83576 | 0.911588412 |
| 30  | 0.85  | 0.83628 | 0.90959041 |
| 35  | 0.845  | 0.83652 | 0.908591409 |

![Simple Tokenizer Accuracy]
(/results/Simple_Tokenizer_Accuracy.png)

###Advanced Tokenizer
| Alpha  | Review Polarity Accuracy | IMDB Accuracy |
|---|:---:|:---:|
| 1  | 0.805  | 0.83364 |
 5  | 0.8275  | 0.84168 |
| 10  | 0.8275  | 0.8436 |
| 15  | 0.83  | 0.84436 |
| 20  | 0.8425  | 0.84496 |
| 25  | 0.8325  | 0.84532 |
| 30  | 0.8425  | 0.84548 |
| 35  | 0.8275  | 0.84592 |

![Advanced Tokenizer Accuracy]
(/results/Advanced_Tokenizer_Accuracy.png)

###Bigram Tokenizer
| Alpha  | Review Polarity Accuracy | IMDB Accuracy |
|---|:---:|:---:|
| 1  | 0.8425  | 0.8624 |
| 5  | 0.86  | 0.86788 |