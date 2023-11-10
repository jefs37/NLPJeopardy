from flask import Flask, request, render_template
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)

app = Flask(__name__)

df = pd.read_csv('jeopardy_full_questions_with_topics.csv')
categories = (df['topic_name'].unique())
categories = categories[1:]

current_category = categories[0]

@app.route('/')
def index():
    return render_template('index.html', current_category = categories[0])

@app.route('/learn', methods = ['POST', 'GET'])
def learn():
    if request.method == 'POST':
        current_category = request.form.get('learn_category')
        if not current_category:
            current_category = categories[0]
        cat_df = df[df['topic_name'] == current_category]
        random_sample = cat_df.sample(n=1)
        current_question = random_sample['question'].values[0]
        current_answer = random_sample['answer'].values[0]
        print(random_sample)
        return render_template('learn.html',
                               categories = categories,
                               current_category = current_category,
                               current_question = current_question,
                               current_answer = current_answer)
    else:
        return render_template('learn.html')
    
@app.route('/game', methods = ['POST', 'GET'])
def game():
    return render_template('game.html')

if __name__ == '__main__':
    app.run(debug = True)