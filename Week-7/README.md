# Week 7
---
In this week, we'll be learning about Natural Language Processing(NLP) which can help computers analyze text easily i.e detect spam emails, autocorrect. We’ll see how NLP tasks are carried out for understanding human language.
## What is Natural Language Processing ?
NLP is a field in machine learning with the ability of a computer to understand, analyze, manipulate, and potentially generate human language.
### NLP in Real Life
- Information Retrieval(Google finds relevant and similar results).
- Information Extraction(Gmail structures events from emails).
- Machine Translation(Google Translate translates language from one language to another).
- Text Simplification(Rewordify simplifies the meaning of sentences). Shashi Tharoor tweets could be used(pun intended).
- Sentiment Analysis(Hater News gives us the sentiment of the user).
- Text Summarization(Smmry or Reddit’s autotldr gives a summary of sentences).
- Spam Filter(Gmail filters spam emails separately).
- Auto-Predict(Google Search predicts user search results).
- Auto-Correct(Google Keyboard and Grammarly correct words otherwise spelled wrong).
- Speech Recognition(Google WebSpeech or Vocalware).
- Question Answering(IBM Watson’s answers to a query).
- Natural Language Generation(Generation of text from image or video data.)
### (Natural Language Toolkit)NLTK: 
NLTK is a popular open-source package in Python. Rather than building all tools from scratch, NLTK provides all common NLP Tasks. We'll be using this library for our purpose in this week of course.
#### Installing NLTK
Type _!pip install nltk_ in the Jupyter Notebook or if it doesn’t work in cmd type _conda install -c conda-forge nltk_. This should work in most cases.

## Reading and Exploring Dataset
### Reading in text data & why do we need to clean the text?
While reading data, we get data in the structured or unstructured format. A structured format has a well-defined pattern whereas unstructured data has no proper structure. In between the 2 structures, we have a semi-structured format which is a comparably better structured than unstructured format.
## Pre-processing Data
Cleaning up the text data is necessary to highlight attributes that we’re going to want our machine learning system to pick up on. Cleaning (or pre-processing) the data typically consists of a number of steps:
### 1. Remove punctuation
Punctuation can provide grammatical context to a sentence which supports our understanding. But for our vectorizer which counts the number of words and not the context, it does not add value, so we remove all special characters. eg: How are you?->How are you
### 2.Tokenization
Tokenizing separates text into units such as sentences or words. It gives structure to previously unstructured text. eg: Plata o Plomo-> ‘Plata’,’o’,’Plomo’.
### 3. Remove stopwords
Stopwords are common words that will likely appear in any text. They don’t tell us much about our data so we remove them. eg: silver or lead is fine for me-> silver, lead, fine.
### Preprocessing Data: Stemming
Stemming helps reduce a word to its stem form. It often makes sense to treat related words in the same way. It removes suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. It reduces the corpus of words but often the actual words get neglected. eg: Entitling,Entitled->Entitl
Note: Some search engines treat words with the same stem as synonyms.
### Preprocessing Data: Lemmatizing
Lemmatizing derives the canonical form (‘lemma’) of a word. i.e the root form. It is better than stemming as it uses a dictionary-based approach i.e a morphological analysis to the root word.eg: Entitling, Entitled->Entitle
<br/>
In Short, Stemming is typically faster as it simply chops off the end of the word, without understanding the context of the word. Lemmatizing is slower and more accurate as it takes an informed analysis with the context of the word in mind.
### Vectorizing Data
Vectorizing is the process of encoding text as integers i.e. numeric form to create feature vectors so that machine learning algorithms can understand our data.
### Vectorizing Data: Bag-Of-Words
Bag of Words (BoW) or CountVectorizer describes the presence of words within the text data. It gives a result of 1 if present in the sentence and 0 if not present. It, therefore, creates a bag of words with a document-matrix count in each text document.
### Vectorizing Data: N-Grams
N-grams are simply all combinations of adjacent words or letters of length n that we can find in our source text. Ngrams with n=1 are called unigrams. Similarly, bigrams (n=2), trigrams (n=3) and so on can also be used. 
Unigrams usually don’t contain much information as compared to bigrams and trigrams. The basic principle behind n-grams is that they capture the letter or word is likely to follow the given word. The longer the n-gram (higher n), the more context you have to work with.
### Vectorizing Data: TF-IDF
It computes “relative frequency” that a word appears in a document compared to its frequency across all documents. It is more useful than “term frequency” for identifying “important” words in each document (high frequency in that document, low frequency in other documents).
### Feature Engineering: Feature Creation
Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. It is like an art as it requires domain knowledge and it can tough to create features, but it can be fruitful for ML algorithm to predict results as they can be related to the prediction.
### Building ML Classifiers: Model selection
We use an ensemble method of machine learning where multiple models are used and their combination produces better results than a single model(Support Vector Machine/Naive Bayes). Ensemble methods are the first choice for many Kaggle Competitions. Random Forest i.e multiple random decision trees are constructed and the aggregates of each tree are used for the final prediction. It can be used for classification as well as regression problems. It follows a bagging strategy where randomly.
Grid-search: It exhaustively searches overall parameter combinations in a given grid to determine the best model.
Cross-validation: It divides a data set into k subsets and repeat the method k times where a different subset is used as the test set i.e in each iteration.
## Conclusion
With NLP there are many different tools and methods you can use. It is worth taking the time to understand how the NLP libraries preprocess text in different ways, ensuring you choose the best method for your task. Making use of pipelines with the preprocessing and modeling steps help to streamline the workflow while cleaning up the code.
<br/>
<br/>
Now, as we learnt above NLP and soon we'll apply it but before that, let us learn about one of the essential field on which constant researches are going on. Imagine that you have to create a machine that can perform a specific action without any assistance from humans. But, accomplishing such real-world tasks by a machine is a complex process. Thus, you need a technique that allows the machine to learn by itself. This technique is reinforcement learning.
## Reinforcement Learning
Reinforcement learning is an area of Data Science. It is about taking suitable action to maximize reward in a particular situation. It is employed by various software and machines to find the best possible behavior or path it should take in a specific situation. Reinforcement learning differs from the supervised learning in a way that in supervised learning the training data has the answer key with it so the model is trained with the correct answer itself whereas in reinforcement learning, there is no answer but the reinforcement agent decides what to do to perform the given task. In the absence of a training dataset, it is bound to learn from its experience.
### Main points in Reinforcement learning –
- Input: The input should be an initial state from which the model will start
- Output: There are many possible output as there are variety of solution to a particular problem
- Training: The training is based upon the input, The model will return a state and the user will decide to reward or punish the model based on its output.
- The model keeps continues to learn.
- The best solution is decided based on the maximum reward.
### Various Practical applications of Reinforcement Learning –
- RL can be used in robotics for industrial automation.
- RL can be used in machine learning and data processing
- RL can be used to create training systems that provide custom instruction and materials according to the requirement of students.
### RL can be used in large environments in the following situations:
- A model of the environment is known, but an analytic solution is not available.
- Only a simulation model of the environment is given (the subject of simulation-based optimization)
- The only way to collect information about the environment is to interact with it.
## Assignment
---
For this week assignment, you'll be using the above-discussed sections on NLP combined to build a Spam-Ham Classifier. You will need to download the dataset '[SMSSpamCollection.tsv](https://github.com/ShivamModi09/Edunnovate-Technologies-Data-Science/blob/main/Week-7/SMSSpamCollection.tsv)' under Week-7 folder to use it for your assignment. You can take help of NLTK's [documentation](https://www.nltk.org/).
Solution to the Assignment is also under Week-7 folder with name 'NLP.ipynb', refer to it after trying the assignment by your own.
