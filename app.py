from flask import Flask, session, request, redirect, url_for, render_template, jsonify
import pandas as pd
import numpy as np
import sys
import game_mode_functions
from threading import Thread  
import time
import os
pd.set_option('display.max_colwidth', None)

app = Flask(__name__)
app.config['first_answer'] = "The \"coq\" in coq au vin"
app.config['user_question'] = "chicken"
app.config['score'] = 0

global current_category, question,first_answer,score,number_of_rounds, is_correct, cat_df

df = pd.read_csv('jeopardy_full_questions_with_topics.csv')
categories = np.array(df['topic_name'].unique())
categories = categories[1:]
clue_values = np.array(df['clue_value'].unique())
clue_indexes = np.argsort(clue_values)
clue_values = clue_values[clue_indexes]
current_category = categories[0]
current_value = clue_values[0]

score = 0
first_answer = "Your first answer here"
current_category = categories[0]
question =''
number_of_rounds=1
df_preprocessed, tf_ans,normalize_corpus = game_mode_functions.GameMode_PreprocessText(df)

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

@app.route('/get_input', methods=['POST'])  
def get_input():  
    text_input = request.form['input']  
    user_question = text_input
    print(user_question)
    return jsonify({'status':'success'}), 200  

@app.route('/game', methods = ['POST', 'GET'])

def game():
    first_answer = "The ""\coq\" in coq au vin"
    user_question = "chicken"
    current_category = categories[0]

    user_question = "Your question"
    message_to_users=''
    display_text = False
    if request.method == 'POST' and request.form.get('submit_button') == 'Select Category': 
        current_category = request.form.get('selected_option')

        print("selected category:",current_category)
        if not current_category:
            current_category = categories[0]
        cat_df = df_preprocessed[df_preprocessed['topic_name'] == current_category]
        
        first_answer = game_mode_functions.GetFirstAnswer(cat_df)
        print("first_answer: ", first_answer," from category:",current_category)

        df_user = pd.DataFrame()
        df_user['current_category'] = [current_category]
        df_user['first_answer'] = [first_answer]
        df_user.to_csv("df_user.csv")


        return render_template('game.html',
                            # display_text = display_text,
                            # message_to_users = message_to_users,
                            current_category = current_category,
                            current_value = int(current_value),
                            first_answer = first_answer)
    
    else:
        return render_template('game.html')

@app.route('/finalscore', methods=['GET', 'POST'])  
def finalscore():  
    df_user = pd.read_csv("df_user.csv")
    final_score = df_user['score'].iloc[0]
    first_answer = df_user['first_answer'].iloc[0]
    current_category = df_user['current_category'].iloc[0]

    print("Score:",score)
    
    if request.method == 'POST' :
        if len(df_user)>1:
            final_score = np.sum(df_user['score'])
        if request.form.get('submit_button') == 'Quit':
            os.remove("df_user.csv")  

    return render_template('finalscore.html', final_score = final_score)  

@app.route('/get_text', methods=['POST','GET'])  
def get_text():  
    # questions_list=[]
    # score_list=[]

    question = request.json.get('text', None)  
    print("You entered question:", question)
    df_user = pd.read_csv("df_user.csv")
    # questions_list =df_user['question'].tolist()
    # questions_list.append(question)
    # df_user['question'] = questions_list#[question ]
    df_user['question'] = [question ]

    df_user.to_csv("df_user.csv")

    score, is_correct = game_mode_functions.CheckQuestion(normalize_corpus)
    df_user['score'] = [score]
    # score_list =df_user['score'].tolist()

    # score_list.append(score)
    # df_user['score']=score_list
    df_user.to_csv("df_user.csv")
    
    return jsonify({"message": "success"})  

if __name__ == '__main__':
    app.run(debug = True)