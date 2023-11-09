from flask import Flask, request, render_template
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)

app = Flask(__name__)

df = pd.read_csv('jeopardy_full_questions_with_topics.csv')

np.random.seed(37)
rand_index = np.random.randint(0, df.shape[0], 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learn', methods = ['POST', 'GET'])
def learn():
    if request.method == 'POST':
        rand_index = np.random.randint(0, df.shape[0], 1)
        current_question = df.loc[rand_index]['question']
        current_answer = df.loc[rand_index]['answer']
        return render_template('learn.html',
                               current_question = current_question.values[0],
                               current_answer = current_answer.values[0])
    else:
        return render_template('learn.html')
    
@app.route('/game', methods = ['POST', 'GET'])
def game():
    return render_template('game.html')

if __name__ == '__main__':
    app.run(debug = True)