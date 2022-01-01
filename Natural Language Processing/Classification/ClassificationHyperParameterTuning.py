#Author: Boris Kundu
#Hyper parameter tuning using GridSearchCV

#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import html
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings

#Read data
df = pd.read_csv('eclipse_jdt.csv')

#Disable warnings
warnings.filterwarnings("ignore")
#Check sample
print(f'Sample Data:{df.sample(1).T}')
#Check info
print(f'\nDataset Info:\n{df.info}')
#Check stats
print(f'\nData Stats:\n{df.describe()}')

#Check priority counts
df['Priority'].value_counts().sort_index().plot(kind='bar')
plt.show()

#Select features of use
df = df[['Title','Description','Priority']]
# Create text from Title and Description 
df['text'] = df['Title'] + ' ' + df['Description']
# Drop unused data
df = df.drop(columns=['Title','Description'])
# Display columns
print(f'Columns:{df.columns}')

#Function to clean text data
def clean(text):
    # convert html escapes like &amp; to characters.
    text = html.unescape(text)
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

#Check for umpurtity. Lower is better
RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')

#Returns share of suspicious characters
def impurity(text, min_len=10):
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)

print(f'Original Priority Counts:\n{df["Priority"].value_counts()}')
# Filter bug reports with priority P3 and sample 3500 rows from it
df_sampleP3 = df[df['Priority'] == 'P3'].sample(n=3500)
# Create a separate DataFrame containing all other bug reports
df_sampleRest = df[df['Priority'] != 'P3']
# Concatenate the two DataFrame to create the new balanced bug reports dataset
df_balanced = pd.concat([df_sampleRest, df_sampleP3])
# Check the status of the class imbalance
print(f'\nBalanced Priority Counts:\n{df_balanced["Priority"].value_counts()}')

# Loading the balanced DataFrame
df = df_balanced[['text', 'Priority']]
# Drop missing data
df = df.dropna()

#Clean data
df['text'] = df['text'].apply(clean)
#Keep data of length > 50 characters
df = df[df['text'].str.len() > 50]

#Split data
X_train, X_test, Y_train, Y_test = train_test_split(df['text'],
                                                    df['Priority'],
                                                    test_size=0.3,
                                                    random_state=101,
                                                    stratify=df['Priority'])
print('Size of Training Data ', X_train.shape[0])
print('Size of Test Data ', X_test.shape[0])

#Transform data to tfidf
tfidf = TfidfVectorizer(min_df = 10, ngram_range=(1,2), stop_words="english")
X_train_tf = tfidf.fit_transform(X_train)

#Train and fit model
model1 = LinearSVC(random_state=101, tol=1e-5)
model1.fit(X_train_tf, Y_train)

X_test_tf = tfidf.transform(X_test)
#Predict priorty class
Y_pred = model1.predict(X_test_tf)
print ('Accuracy Score - ',accuracy_score(Y_test, Y_pred))

#Plot confusion matrix
plot_confusion_matrix(model1,X_test_tf, Y_test, values_format='d', cmap = plt.cm.Blues)
plt.title('Original model - Confusion Matrix')
plt.show()

#Display classification report
print(f'\nClassification Report:\n{classification_report(Y_test, Y_pred)}')

# Create a DataFrame combining the Title and Description,
# Actual and Predicted values that we can explore
frame = { 'text': X_test, 'actual': Y_test, 'predicted': Y_pred }
result = pd.DataFrame(frame)

#Check invalid predictions
result[((result['actual'] == 'P1') | (result['actual'] == 'P2')) &
        (result['actual'] != result['predicted'])].sample(2)

# Perform K-Fold Cross Validation
# Vectorization
tfidf = TfidfVectorizer(min_df = 10, ngram_range=(1,2), stop_words="english")
df_tf = tfidf.fit_transform(df['text']).toarray()
# Cross Validation with 5 folds
scores = cross_val_score(estimator=model1,
                        X=df_tf,
                        y=df['Priority'],
                        cv=5)
print ("Validation scores from each iteration of the cross validation ", scores)
print ("Mean value across of validation scores ", scores.mean())
print ("Standard deviation of validation scores ", scores.std())

# Form training pipeline for GridSearchCV
training_pipeline = Pipeline(steps = [('tfidf', TfidfVectorizer(stop_words = "english")), ('model', LinearSVC(random_state = 101, tol = 1e-5))])

# Defne parameters for the grid
grid_param = [
                {
                'tfidf__min_df': [5, 10],
                'tfidf__ngram_range': [(1,3), (1,4)],
                'model__penalty': ['l2'],
                'model__loss': ['hinge'],
                'model__max_iter': [5]
                }, 
                {
                'tfidf__min_df': [5, 10],
                'tfidf__ngram_range': [(1,3), (1,4)],
                'model__C': [1,5],
                'model__tol': [1e-2, 1e-3],
                'model__max_iter': [5]
                }
            ]

# Define and rain GridSearchCV using he pipeline and parameters defined above 
gridSearchProcessor = GridSearchCV(estimator = training_pipeline, param_grid=grid_param, cv=5)
gridSearchProcessor.fit(df['text'], df['Priority'])

# Hyper-parameter tuning using GridSearchCV for text classification.
# Get best params
best_params = gridSearchProcessor.best_params_
print("Best alpha parameter identified by grid search ", best_params)

#Get best result
best_result = gridSearchProcessor.best_score_
print("Best result identified by grid search ", best_result)

gridsearch_results = pd.DataFrame(gridSearchProcessor.cv_results_)
print(f"Top 5 Results:\n{gridsearch_results[['rank_test_score', 'mean_test_score','params']].sort_values(by = ['rank_test_score'])[:5]}")

best_model = gridSearchProcessor.best_estimator_
# Model Evaluation
Y_pred = best_model.predict(X_test)
print('Best Model Accuracy Score:', accuracy_score(Y_test, Y_pred))
print(f'Best Model Classification Report:\n{classification_report(Y_test, Y_pred)}')

#Plot confusion matrix
plot_confusion_matrix(best_model,X_test,Y_test, values_format='d', cmap = plt.cm.Blues)
plt.title('Hyperparameter Tuned Model - Confusion Matrix')
plt.show()