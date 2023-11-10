import pandas as pd
import numpy as np

df = pd.read_csv('jeopardy_full_questions_with_topics.csv')

categories = (df['topic_name'].unique())
categories = categories[1:]
current_category = 'World Talk'
cat_df = df[df['topic_name'] == current_category]

print(cat_df)
print(df)

rand_index = np.random.randint(0, cat_df.shape[0], 1)
current_question = cat_df.loc[137]['question']

print(current_question)