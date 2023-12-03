import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
global normalize_corpus
stop_words = nltk.corpus.stopwords.words('english')

df = pd.read_csv('jeopardy_full_questions_with_topics.csv')
def normalize_document(doc):
  # lower case and remove special characters\whitespaces
  doc = str(doc)
  doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
  doc = doc.lower()
  doc = doc.strip()
  # tokenize document
  tokens = nltk.word_tokenize(doc)
  # filter stopwords out of document
  filtered_tokens = [token for token in tokens if token not in stop_words]
  # re-create document from filtered tokens
  doc = ' '.join(filtered_tokens)
  return doc

def GameMode_PreprocessText(df):
  normalize_corpus = np.vectorize(normalize_document)
  norm_corpus_ans = normalize_corpus(list(df['answer']))
  norm_corpus_q = normalize_corpus(list(df['question']))
  df['normalized_answer'] = norm_corpus_ans
  df['normalized_question'] = norm_corpus_q

  tf_ans = TfidfVectorizer(ngram_range=(1, 3), min_df=2)
  tfidf_matrix_ans = tf_ans.fit_transform(norm_corpus_ans)
  tf_q= TfidfVectorizer(ngram_range=(1, 2), min_df=2)
  tfidf_matrix_q = tf_q.fit_transform(norm_corpus_q)
  return df, tf_ans,normalize_corpus

#### Compute Pairwise Document Similarity among pairs of answers within each category
#### Function to get next recommended answer based on previous answer

from sklearn.metrics.pairwise import cosine_similarity
# def GetAnswerSimilarity(df, q_category,prev_answer):
def RecommendNextQuestion(df_cat):
  print("entered reco model")
  df_user = pd.read_csv("df_user.csv")

  if len(df_user) <=0:
    return GetFirstAnswer(df_cat)
  else:
    prev_answer=df_user['first_answer'].iloc[-1]
    df_features, tf_ans,normalize_corpus = GameMode_PreprocessText(df)

    #Filter dataframe for answers that belong to the category selected by the user
    # df_cat = df[df['topic_name']==q_category]
    ans_list = df_cat['normalized_answer'].values
    normalized_prev_answer = df_cat['answer'].tolist().index(prev_answer)
    # print("Normalized answer:",ans_list[normalized_prev_answer])
    ans_idx = np.where(ans_list == ans_list[normalized_prev_answer])[0][0]

    tfidf_matrix_ans = tf_ans.fit_transform(df_cat['normalized_answer'])
    doc_sim_ans = cosine_similarity(tfidf_matrix_ans)
    doc_sim_df = pd.DataFrame(doc_sim_ans)
    ans_similarities = doc_sim_df.iloc[ans_idx].values
    similar_ans_idxs = np.argsort(-ans_similarities)[1:6]
    similar_ans = ans_list[similar_ans_idxs]
    non_norm_ans =  df_cat['answer'].values
    similar_ans =non_norm_ans[similar_ans_idxs]
    return similar_ans[0]

def GetNormalizedCorpus():
  return normalize_corpus

#Get random answer as the first answer to the user in game mode
def GetFirstAnswer(df_cat):

  import random
  random.seed(10)

  # df_cat = df[df['topic_name']==q_category]
  random_number = random.randint(0, len(df_cat))
  non_norm_ans =  df_cat['answer'].values
  random_ans =non_norm_ans[random_number]
  return random_ans

#Get question as the first question to the user in game mode
def GetFirstQuestion(df_cat,answer):
  # df_cat = df[df['topic_name']==q_category]
  non_norm_ans =  df_cat['answer'].tolist()
  non_norm_q =  df_cat['question'].values
  # print(non_norm_ans,answer,non_norm_ans.index(answer))
  random_q =non_norm_q[non_norm_ans.index(answer)]
  return random_q

### Function to evaluate question entered by user for calculating the score if it's correct or not
def TextSimilarity(text1, text2):
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import cosine_similarity

  
  vectorizer = TfidfVectorizer().fit_transform([text1, text2])
  vectors = vectorizer.toarray()

  sim_score = cosine_similarity(vectors)
  # print(sim_score[0][1])

  return sim_score[0][1]

def CheckQuestion(normalize_corpus):
  print("entered CheckQuestion")
  df_user = pd.read_csv("df_user.csv")
  answer = df_user['first_answer'].iloc[0]
  user_q = df_user['question'].iloc[0]
  current_category = df_user['current_category'].iloc[0]

  df_cat = df[df['topic_name'] == current_category]
  # normalize_corpus = GetNormalizedCorpus()

  norm_corpus_q = normalize_corpus(user_q)
  print("Normalized question:",norm_corpus_q)
  #Get original question for given answer
  ans_index = df_cat['answer'].tolist().index(answer)
  print(df_cat['answer'].iloc[ans_index])
  real_question = df_cat['question'].iloc[ans_index]
  print("Real question is:",real_question)
  clue_values = df_cat['clue_value'].tolist()
  is_correct = True
  if( (user_q == real_question ) or TextSimilarity(user_q, real_question)) > 0.5:
    print("You are right!!! You won: $",df_cat['clue_value'].iloc[ans_index])
    score = clue_values[ans_index]
  else:
    score =0
    print("Sorry, that's incorrect", score)
    is_correct = False

  print("So far you won: $",score)
  return score

