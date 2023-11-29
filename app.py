from flask import Flask, request, render_template
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)

app = Flask(__name__)

df = pd.read_csv('jeopardy_full_questions_with_topics.csv')
categories = np.array(df['topic_name'].unique())
categories = categories[1:]
clue_values = np.array(df['clue_value'].unique())
clue_indexes = np.argsort(clue_values)
clue_values = clue_values[clue_indexes]
current_category = categories[0]
current_value = clue_values[0]

@app.route('/')
def index():
    return render_template('index.html', current_category = categories[0])

@app.route('/wordclouds')
def wordclouds():
    return render_template('wordclouds.html')

@app.route('/learn', methods = ['POST', 'GET'])
def learn():
    if request.method == 'POST':
        current_category = request.form.get('learn_category')
        current_value = request.form.get('learn_value')
        if not current_category:
            current_category = categories[0]
            current_value = clue_values[0]
        cat_df = df[df['topic_name'] == current_category]
        if int(current_value) == 0:
            final_df = cat_df
        else:
            final_df = cat_df[cat_df['clue_value'] == int(current_value)]
        random_sample = final_df.sample(n=1)

        current_question = random_sample['question'].values[0]
        current_answer = random_sample['answer'].values[0]

        #removing backslashes
        current_question = current_question.replace("\\", "")
        current_answer = current_answer.replace("\\", "")

        return render_template('learn.html',
                               categories = categories,
                               clue_values = clue_values,
                               current_category = current_category,
                               current_value = int(current_value),
                               current_question = current_question,
                               current_answer = current_answer)
    else:
        return render_template('learn.html')

@app.route('/game', methods = ['POST', 'GET'])
def game():
    return render_template('game.html')

if __name__ == '__main__':
    app.run(debug = True)