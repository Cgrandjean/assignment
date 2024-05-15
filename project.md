# Hate Speech Detection Project

## 1. Data Exploration and Preprocessing
### Exploration:

    I started by exploring the dataset to understand the distribution of the training and test sets.I verified the datasets were balanced. that hte word  distributions were similar for both training and test set.I also checked the length of documents. They seem to more or less match.I also compared the distributions based on categories. 

    Possible improvements: Using this study of distributions I could generate synthetic augmented data.Focusing on keeping the words that are moost representative for each categories and sligthly modifying the neighboring ones 

### Preprocessing:
    The preprocessing consisted simply in loading metadata and tokenizing the documents.
    
    Possible improvements: A part of my preprocessing is done in 02_01_train_nb, i should have included it in the preprocessing notebook,it was just faster for the pipelining of my model.I could also have filtered some outliers(long documents).If more time would be allocated to the project it should be explored

## 2. Training the Naive Bayes Model
    I wanted to implement a fast simple model to start with .So i decided to train a Naive Bayes model, i removed stop words using nltk and use calssical range of hyperparameters.I tried few configurations and kept the most reasonable one.

    Possible improvements:I should have done cross validation (as the dataset is relatively small) and select hyperparameters in a better way but i lacked time.

## 3. Training BERT Model
    As transformers are the most performing class of model in NLP, i wanted to try implementing a Simple BERT on that topic.The BERT model was fine-tuned over a few epochs with relatively and selected based on eval_loss metric.

    Possible improvements: My BERT model was simply fine tuned on the data, but its pre-training was maybe not suited for this kind of language.I could have used specific version of BERT trained on hate speech to match more this distributiion of words.I could also have used another metric to select the best model(Roc_auc seems to grant better performance overall).I also did not have time to play with Hyperparameters

## 4. Evaluation
    I created a notebook to compare the performance of the Naive Bayes and BERT models. Key metrics such as accuracy, precision, recall, F1-score, and ROC AUC were used to evaluate the models. Unsurprinsingly BERT surpassed Naive bayes.

## 5. Model Deployment
    I simply deployed my model using FASTAPI.I also pushed my BERT model to the hub as its a bit heavy and is clumbersome for the repo.

## Conclusion
    This project successfully developed a model for hate speech detection, starting from data exploration and preprocessing to model training, evaluation, and deployment. BERT model demonstrated superior performance(at the cost of longer inference time,though its still really reasonable). and the deployed model is now ready for real-time inference.
    
    Curl command examples:
                 command: curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' -d '{"text": "those people are stupid"}'
                  result: {"text":"those people are stupid","predictions":[{"label":"Hate","score":0.9871842265129089}]}
                 command: curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' -d '{"text": "I like this movie"}'
                  result: {"text":"I like this movie","predictions":[{"label":"Nohate","score":0.9973760843276978}]}