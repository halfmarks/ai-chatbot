#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications
"""
#######################################################
# Initialise Wikipedia agent
#######################################################

import wikipedia
import numpy as np
import os
import sys
import json, requests
import csv
#import spotipy.util as util
import aiml
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#username for spotify



#try:
#     token = util.prompt_for_user_token[username]
#except:
#     os.remove(f".cache-{username}")
#     token = util.prompt_for_user_token(username)
#
##creating spotify object
#spotifyObject = spotipy.Spotify(auth=token)

#######################################################
# Initialise weather agent
#######################################################

# insert your personal OpenWeathermap API key here if you have one, and want to use this feature
APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f"
file_name = 'sampleQA.csv'
cosine_threshold = 0.3
#######################################################
#  Initialise AIML agent
#######################################################



# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")


print("Welcome to this chat bot. Please feel free to ask questions from me!")

# Main loop


while True:
    # get user input
    try:
         userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break

#    my_data = np.genfromtxt('sampleQA.csv', delimiter=';')

    
    with open(file_name) as csv_file:
#        csv_reader = csv.reader(csv_file, delimiter=';')
        questions = list(csv.reader(csv_file, delimiter=';'))

    questions = np.array(questions)

    count = 0
    for row in questions:
        data = [userInput, row[0]]
       
#       from sklearn.feature_extraction.text import TfidfVectorizer
        Tfidf_vect = TfidfVectorizer()
        vector_matrix = Tfidf_vect.fit_transform(data)

        tokens = Tfidf_vect.get_feature_names()
        
        
        cosine_similarity_matrix = cosine_similarity(vector_matrix)
      
        if cosine_similarity_matrix[0,1] >= cosine_threshold:
            print (row[1])
            break
        else:
             count += 1
        
    responseAgent = 'aiml'
    # pre-process user input and determine response agent (if needed)
    # activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    # post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=False)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")
        elif cmd == 2:
            succeeded = False
            api_url = r"http://api.openweathermap.org/data/2.5/weather?q="
            response = requests.get(api_url + params[1] + r"&units=metric&APPID=" + APIkey)
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json:
                    t = response_json['main']['temp']
                    tmi = response_json['main']['temp_min']
                    tma = response_json['main']['temp_max']
                    hum = response_json['main']['humidity']
                    wsp = response_json['wind']['speed']
                    wdir = response_json['wind']['deg']
                    conditions = response_json['weather'][0]['description']
                    print("The temperature is", t, "°C, varying between", tmi, "and", tma, "at the moment, humidity is",
                          hum, "%, wind speed ", wsp, "m/s,", conditions)
                    succeeded = True
            if not succeeded:
                print("Sorry, I could not resolve the location you gave me.")


        elif cmd == 4:
             from lyricsgenius import Genius
             
             payload={
                       'genius_client_id' : '3HkmTIjUzV_rfcEanEOqqMNaxujiP2pVGBOIUJ5bIAYg-Pod8dGFNztkEDfWaO1c',
                       'genius_secret_id' : 'YhMhZq8q_negezQ9tB5v8UcRhYfXYFx0P3Z6gu9lZP9SlFPaOnfn1xSAaCTAEHS_ecZ_on2d8vuQikgu8uyzgQ',
                       'genius_client_access_token' : 'qpyLvNHylf1k4d0z5kLiZ5R7g6jU_KmYDfKaNR1x5OI81uHIcRoAeeQz05P5S2Ac'}

             base_url = 'https://api.genius.com/'
             r = requests.get(base_url, params=payload)

             client_access_token = ''
             token = 'Bearer{}'.format(client_access_token)
             genius = Genius(token)
             artist = genius.search_artist(params[1], sort="title")
             
             song = genius.search_song(params[0], artist.name)
             song = artist.song(params[0])
             
             print(song.lyrics)
             

        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer)
