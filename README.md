<<<<<<< Updated upstream
=======
# insincere-word-detection
>>>>>>> Stashed changes
# insincere-question-detection


# Introduction 
There are plenty of online platforms, such as Quora and StackExchange, that allow people to share their knowledge as well as to learn from each other. However, one issue that these websites now face is how to distinguish and remove the toxic and insincere questions and comments. This project aims to develop a model that can identify insincere questions. Some typical insincere questions include questions having a non-natural tone, grounded on false premises, disparaging or inflammatory, or using sexual content for shock values. By tackling those insincere questions and toxic contents, we can contribute to keep the platform a safe and comfortable place for people to share information. 

# Data and methodology 
The Quora insincere question set can be downloaded here: https://www.kaggle.com/c/quora-insincere-questions-classification/data. The training data set includes the actual questions that were asked on Quora(“question_text”), and a label(“target”) that indicates whether each question is an insincere question (“target”=1) or not (“target”=0). The labels have some noises and are not perfect. The testing set contains another set of questions that requires us to classify as insincere or not. 
Since we do not have access to the ground truth label of the testing file, we will randomly split the training dataset for training and validation sets. 
We specifically study three classification methods, the logistic regression, the naive Bayes classifier and the bidirectional Long-Short term memory, which is a type of recurrent neural network.
We preprocess the raw text data to make it consistent through operations including tokenization, stemming uncapitalization, removing numbers and adding tags to all capitalized words. 
We treat logistic regression and naive bayes as two baselines. We transform each of the questions to a vector using TF-IDF transformation and apply the two basic classification methodologies to the transformed vectors. 
Considering the BiLSTM classifier, we perform several experiments on different pretrained embedding layers as well as nonpretrained one. 



