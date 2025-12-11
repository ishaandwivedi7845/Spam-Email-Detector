# Spam-Email-Detector
This is my first machine learning project that i have built on my own. it automatically detects spam and not spam(ham) messages by a voting classifier that gives upto 97% accuracy with no false positives
# Process
i used the following libraries from sklearn-
-sklearn.metrics
    to generate accuracy and confusion matrix and assess the accuracy of the overall model
-sklearn.svm,sklearn.linear_model and sklearn.ensemble- from these libraries i took binary classifying models(RandomForestClassifier,logistic regression and support vector classifier)
-sklearn.ensemble-from here i took the voting classifier
-sklearn.model_selection-from here i took the train_test_split to train my model on 80% of the data and test it on the rest
-sklearn.feature_extraction.text- from here i took TfidfVectorizer to convert texts into vectors and add weight to it
-sklearn.preprocessing- from here i took label_encoder to encode spam and ham to 0 and 1 , that can be processed by the models

    

