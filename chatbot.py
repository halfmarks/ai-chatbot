#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#######################################################
# Initialise Wikipedia agent
#######################################################

import csv
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import requests
import pandas
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import aiml
import numpy as np
import wikipedia
import tensorflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
import azure.cognitiveservices.speech as speechsdk

from msrest.authentication import CognitiveServicesCredentials

# Keys for azure portal
cog_key = '1073fe9a6edc4369a5ea5248723c36ea'
credential = AzureKeyCredential(cog_key)
cog_endpoint = 'https://cognitiveserviceai.cognitiveservices.azure.com/'
cog_region = 'uksouth'

text_analytics_client = TextAnalyticsClient(endpoint=cog_endpoint,
                                            credentials=AzureKeyCredential(cog_key))

# input and Chatbot Language
inputLanguageCode = ""
chatbot_language = "en"

#######################################################
# Initialise weather agent
#######################################################
# insert your personal OpenWeathermap API key here if you have one, and want to use this feature
APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f"
cosine_threshold = 0.3

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


def authenticate_client():
    ta_credential = AzureKeyCredential('1073fe9a6edc4369a5ea5248723c36ea')
    text_analytics_client = TextAnalyticsClient(
        endpoint='https://cognitiveserviceai.cognitiveservices.azure.com/',
        credential=ta_credential)
    return text_analytics_client


client = authenticate_client()


def language_detection(client):
    try:
        documents = [userInput]
        response = client.detect_language(documents=documents, country_hint='us')[0]
        #  print("Language: ", response.primary_language.name)
        theLang = response.primary_language.iso6391_name
        #  print(thelang)
        return theLang

    except Exception as err:
        print("Encountered exception. {}".format(err))


def lang_code(inputLanguage):
    switcher = {
        'English': 'en',
        'French': "fr",
        'German': 'de',
    }

    return switcher.get(inputLanguage, "nothing")


def translate_text(cog_region, cog_key, text, to_lang='fr', from_lang='en'):
    import requests, uuid, json

    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = path + params

    # Prepare the request headers with Cognitive Services resource key and region
    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region': cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Add the text to be translated to the body
    body = [{
        'text': text
    }]

    # Get the translation
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]["text"]


def translate_response(userText, langCode):
    translated = translate_text(cog_region, cog_key, userText, to_lang=langCode,
                                from_lang='en')

    if audioChoice == "yes":
        speech(translated)

    return translated


def speech(userSpeech):
    # <code>

    # Creates an instance of a speech config with specified subscription key and service region.
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_key, service_region = "6e4c22e7258c4aae99c1ba66bc55b2ef", "uksouth"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Creates a speech synthesizer using the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    # Receives a text from console input.
    # print("Type some text that you want to speak...")
    text = userSpeech

    # Synthesizes the received text to speech.
    # The synthesized speech is expected to be heard on the speaker with this line executed.
    result = speech_synthesizer.speak_text_async(text).get()

    # Checks result.
    # if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
    # print("Speech synthesized to speaker for text [{}]".format(text))
    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
        print("Did you update the subscription info?")


# delete print('Ready to use cognitive services in {} using key {}'.format(cog_region, cog_key))

# Knowledge base logic
read_expr = Expression.fromstring
kb = []
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
# Checking integrity of kb
artists = ['Kendrick', 'drake']
music = ['Rap', 'Jazz']


def prepare(filepath):
    img = image.load_img(filepath, target_size=(256, 256))
    img_array = image.img_to_array(img)

    img_batch = np.expand_dims(img_array, axis=0)

    # format image into shape required for model
    img_preprocessed = preprocess_input(img_batch)
    # load model that has been trained
    model = keras.models.load_model('genre-cnn.model')
    # prediction = model.predict(img_preprocessed)
    truest = np.argmax(model.predict(img_preprocessed), axis=-1)
    # print(prediction)
    # print(truest)
    # print(int(prediction[0][0]))
    # print(model.predict_classes(img_preprocessed))
    return int(truest[0])

    # IMG_SIZE = 256  # 50 in txt-based
    # img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    # return new_array.reshape(IMG_SIZE, IMG_SIZE, 3) return the image with shaping that TF wants.


expr = read_expr('artist(OVO)')
kbAnswer = ResolutionProver().prove(expr, kb, verbose=False)

if not kbAnswer:
    print("Integrity of kb is fine")
else:
    sys.exit("There is something wrong with knowledge base needs to be fixed before continuing")

kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="mybot-basic.xml")

print("Would you like audio enabled")
audioChoice = input("> ")

print("\nWelcome to this chat bot. Please feel free to ask questions from me!")

# Main loop

while True:

    responseAgent = 'aiml'

    # get user input
    try:
        userInput = input("> ")
        # inputs = []
        # inputs.append(userInput)
        # language_analysis = text_analytics_client.detect_language(documents=inputs)
        # lang = language_analysis.documents

        # input language is the language that the user has entered e.g French
        inputLanguageCode = language_detection(client)
        # print(inputLanguage)
        # the language code for use later
        # langCode = lang_code(inputLanguage)
        # print(langCode)

        # if inputLanguageCode != 'en':
        translation = translate_text(cog_region, cog_key, userInput, to_lang='en',
                                     from_lang=inputLanguageCode)

        userInput = translation


    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break

        # if language_detection_example(client) == "English":

    with open('sampleQA.csv', encoding='utf-8') as csv_file:
        # f = codecs.open(csv_file, "r", "utf-16")
        # csv_reader = csv.reader(csv_file, delimiter=';')
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

        if cosine_similarity_matrix[0, 1] >= cosine_threshold:
            csvTranslation = (row[1])
            print(translate_response(csvTranslation, inputLanguageCode))
            responseAgent = 'csv'
            break
        else:
            count += 1

    # pre-process user input and determine response agent (if needed)
    # activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
        answer = translate_response(answer, inputLanguageCode)
        # if inputLanguageCode != 'en':
        #     translation = translate_text(cog_region, cog_key, answer, to_lang='en',
        #                                  from_lang=inputLanguageCode)

    # post-process the answer for commands
    if responseAgent == 'csv':
        continue
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break

        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=False)
                print(translate_response(wSummary, inputLanguageCode))
            except:
                invalidResponse = ("Sorry, I do not know that. Be more specific!")
                print(translate_response(invalidResponse, inputLanguageCode))
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
                    temperature = "The temperature is " + str(t) + " °C, varying between " + str(tmi) + " and " + str(
                        tma) + \
                                  " at the moment, humidity is " + str(hum) + " %, wind speed " + str(
                        wsp) + " m/s, " + str(conditions)
                    translatedTemp = translate_response(temperature, inputLanguageCode)
                    print(translatedTemp)
                    succeeded = True

            if not succeeded:
                invalidResponse = ("Sorry, I could not resolve the location you gave me.")
                print(translate_response(invalidResponse, inputLanguageCode))

        elif cmd == 4:
            print("Would you like audio enabled. Yes or no?")
            audioChoice = input("> ")

        elif cmd == 5:

            CATEGORIES = ["Jazz", "Rap", "Rock"]

            # root = tk.Tk()
            # root.withdraw()
            #
            # file_path = filedialog.askopenfilename()
            # root.update()
            enterFileName = "enter file name "
            file_name = input(translate_response(enterFileName, inputLanguageCode))
            if os.path.isfile(file_name):
                guess = prepare(file_name)

                if (CATEGORIES[guess] == 'Jazz'):
                    albumAnswer = ("This is a jazz album")
                    print(translate_response(albumAnswer, inputLanguageCode))

                elif (CATEGORIES[guess] == 'Rap'):
                    albumAnswer = ("This is a rap album")
                    print(translate_response(albumAnswer, inputLanguageCode))

                elif (CATEGORIES[guess] == 'Rock'):
                    albumAnswer = ("This is a rock album")
                    print(translate_response(albumAnswer, inputLanguageCode))
                else:
                    invalidResponse = ("I couldn't identify this image")
                    print(translate_response(invalidResponse, inputLanguageCode))

            else:
                noFile = ("Couldn't find this file")
                print(translate_response(noFile, inputLanguageCode))
            # model = keras.models.load_model('genre-cnn.model')
            # prediction_img = str(params[1])

            # prediction = model.predict([prepare(file_path)])
            # prediction = model.predict([prepare(prediction_img)])

        elif cmd == 18:
            leaving = "bye"
            print(translate_response(leaving, inputLanguageCode))
            sys.exit()

        elif cmd == 31:
            # i know that * is *
            object, subject = params[1].split(' is ')
            expr = read_expr(subject + '(' + object + ')')
            # >>> ADD SOME CODES HERE to make sure expr does not contradict
            # with the KB before appending, otherwise show an error message.
            # If this expression returns true then a contradiction was found in kb
            # If false then no contradiction was found
            kbAnswer = ResolutionProver().prove(expr, kb, verbose=False)

            if kbAnswer:
                print("Yes this is true:3 \tI already know this :)")

            else:
                # check for contradiction in logic
                # if this returns true then
                kbAnswer = ResolutionProver().prove(Expression.negate(expr), kb, verbose=False)
                if kbAnswer:
                    contradiction = ("This contradicts what i know")
                    print(translate_response(contradiction, inputLanguageCode))
                else:
                    addToMemory = ('OK, I will remember that', object, 'is a', subject)
                    print(translate_response(addToMemory, inputLanguageCode))
                    kb.append(expr)

        elif cmd == 32:  # if the input pattern is "check that * is *"
            object, subject = params[1].split(' is an ')
            expr = read_expr(subject + '(' + object + ')')
            # Verbose true = print all steps for inference

            kbAnswer = ResolutionProver().prove(expr, kb, verbose=True)
            if kbAnswer:
                confirm = 'Yes this is true.'
                print(translate_response(confirm, inputLanguageCode))

            else:
                maybe = 'It may not be true.'
                print(translate_response(maybe, inputLanguageCode))
                # The opposite expression
                kbAnswer = ResolutionProver().prove(Expression.negate(expr), kb, verbose=True)
                if kbAnswer:
                    incorrect = "This is incorrect"
                    print(translate_response(incorrect, inputLanguageCode))

                else:
                    unsure = "I'm not sure, not in my knowledge"
                # definite response: either "Incorrect" or "Sorry I don't know."

        elif cmd == 34:  # if the input pattern is "does Kendrick make music"
            label, artist = params[1].split(' sign ')
            expr = read_expr('makes(' + artist + ', ' + label + ')' + '& artist(' + artist + ')')
            # Verbose true = print all steps for inference
            kbAnswer = ResolutionProver().prove(expr, kb, verbose=False)
            if kbAnswer:
                print('Correct.' + artist + ' is signed to ' + label)

            else:
                print('It may not be true.')
                kbAnswer = ResolutionProver().prove(Expression.negate(expr), kb, verbose=False)
                # >> This is not an ideal answer.
                # >> ADD SOME CODES HERE to find if expr is false, then give a
                if (kbAnswer):
                    print("This is incorrect")
                else:
                    print("I'm not sure, not in my knowledge")

        elif cmd == 35:  # if the input pattern is "I know that blah signed blahmusic"
            label, artist = params[1].split(' signed ')
            expr = read_expr('makes(' + artist + ', ' + label + ')')
            # Verbose true = print all steps for inference
            kbAnswer = ResolutionProver().prove(expr, kb, verbose=False)
            if kbAnswer:
                print('Correct.' + artist + ' signed to ' + label)

            else:
                print('It may not be true.')
                kbAnswer = ResolutionProver().prove(Expression.negate(expr), kb, verbose=False)
                if (kbAnswer):
                    print("This contradicts what i know")
                else:
                    print('OK, I will remember that', artist, 'is signed to '
                          , label)
                    kb.append(expr)
        # if the input pattern is "Is "Kendrick" a top artist"
        elif cmd == 36:
            artist = params[1]
            expr = read_expr('topArtist(' + artist + ')')
            # Verbose true = print all steps for inference
            kbAnswer = ResolutionProver().prove(expr, kb, verbose=False)
            if kbAnswer:
                print('Correct.' + artist + ' either has a grammy or has gone platinum.')

            else:
                print('It may not be true.')
                kbAnswer = ResolutionProver().prove(Expression.negate(expr), kb, verbose=False)
                # >> This is not an ideal answer.
                # >> ADD SOME CODES HERE to find if expr is false, then give a
                if (kbAnswer):
                    print("This is incorrect but they may be an upcoming artist")
                else:
                    print("I'm not sure, not in my knowledge")

        # I KNOW * IS A TOP ARTIST
        elif cmd == 37:

            object = params[1]
            expr = read_expr('hasGrammy(' + object + ')')
            kbAnswer = ResolutionProver().prove(expr, kb, verbose=False)

            if (kbAnswer):
                print(translate_response("Yes this is true \tI already know this :)",
                                         inputLanguageCode))

            else:
                # check for contradiction in logic
                # if this returns true then
                kbAnswer = ResolutionProver().prove(Expression.negate(expr), kb, verbose=False)
                if (kbAnswer):
                    print(translate_response("This contradicts what i know",
                                             inputLanguageCode))
                else:
                    print(translate_response('OK, I will remember that' + object + 'has a grammy', inputLanguageCode))
                    kb.append(expr)

        # Does "Kendrick" have a Grammy
        elif cmd == 38:
            artist = params[1]
            expr = read_expr('hasGrammy(' + artist + ')')
            # Verbose true = print all steps for inference
            kbAnswer = ResolutionProver().prove(expr, kb, verbose=False)
            if kbAnswer:
                correct = ('Correct. ' + artist + ' has a grammy.')
                print(translate_response(correct, inputLanguageCode))
            else:
                translate_response('It may not be true.', inputLanguageCode)
                kbAnswer = ResolutionProver().prove(Expression.negate(expr), kb, verbose=False)
                # >> This is not an ideal answer.
                # >> ADD SOME CODES HERE to find if expr is false, then give a
                if (kbAnswer):
                    print(translate_response("This is incorrect but they may be an upcoming artist", inputLanguageCode))
                else:
                    print(translate_response("I'm not sure, not in my knowledge", inputLanguageCode))

        elif cmd == 99:
            invalidResponse = ("I did not get that, please try again.")
            translate_response(invalidResponse, inputLanguageCode)
    else:
        print(answer)
