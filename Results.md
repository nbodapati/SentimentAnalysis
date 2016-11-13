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
| Alpha  | PL04 | IMDB Dataset | Subjectivity |
|---|:---:|:---:|:---:|
| 1  | 0.8025  | 0.83256 | 0.8961038961038961 |
| 5  | 0.8225  | 0.84004 | 0.9050949050949051 |
| 10  | 0.83  | 0.844 | 0.9050949050949051 |
| 15  | 0.835  | 0.84448 | 0.9015984015984015 |
| 20  | 0.8425  | 0.84556 | 0.8986013986013986 |
| 25  | 0.8375  | 0.84692 | 0.8966033966033966 |
| 30  | 0.8275  | 0.84772 | 0.8946053946053946 |
| 35  | 0.8075  | 0.84792 | 0.8931068931068931 |


![Advanced Tokenizer Accuracy]
(/results/Advanced_Tokenizer_Accuracy.png)

###Bigram Tokenizer
| Alpha  | Review Polarity Accuracy | IMDB Accuracy |
|---|:---:|:---:|
| 1  | 0.8425  | 0.8624 |
| 5  | 0.86  | 0.86788 |