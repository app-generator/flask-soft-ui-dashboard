# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
from apps.home import blueprint
from flask import render_template
from flask_login import login_required
from jinja2 import TemplateNotFound

from apps.config import API_GENERATOR

load_dotenv('.env')

openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv('OPENAI_API_KEY')


app = Flask(__name__)
CORS(app)

MODEL = "gpt-3.5-turbo"

df=pd.read_csv('apps/home/processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
df.head()

################################################################################
### Step 13
################################################################################





@blueprint.route('/index')
@login_required
def index():
    return render_template('home/index.html', segment='index', API_GENERATOR=len(API_GENERATOR))

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment, API_GENERATOR=len(API_GENERATOR))

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
    
@blueprint.route('/endpoint', methods=['POST'])
def endpoint():
    if request.method=='POST':
        data = request.json
        question = data['query']


        """
        Create a context for a question by finding the most similar context from the dataframe
        """
        max_len=1800
        # Get the embeddings for the question
        q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
        # Get the distances from the embeddings
        df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


        returns = []
        cur_len = 0

        # Sort by distance and add the text to the context until the context is too long
        for i, row in df.sort_values('distances', ascending=True).iterrows():

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            # Else add it to the text that is being returned
            returns.append(row["text"])

        # Return the context
        context = "\n\n###\n\n".join(returns)

        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Es um especialista em todas as leis de Portugal."},
                {"role": "user", "content": f"Contexto: {context}. Pergunta: {question}"},
                #chat history
            ],
            
            temperature=0,            )
        answer = response['choices'][0]['message']['content']
        print(context)
        print(answer)
        return jsonify({'answer': answer})
        

