# -*- coding: utf-8 -*-

import time
from naoqi import ALProxy
from predict_sentiment import predict_sentiment

#virutal robot
# NAO_IP = "127.0.0.1"
# PORT = 59317

#physical robot
NAO_IP = "172.18.16.45"
PORT = 9559

# Create proxies
dialog = ALProxy("ALDialog", NAO_IP, PORT)
memory = ALProxy("ALMemory", NAO_IP, PORT)
tts = ALProxy("ALTextToSpeech", NAO_IP, PORT)

def get_review_text_from_robot(listen_duration):
    try:
        # Set NAO's listening language
        dialog.setLanguage("English")

        # Clear the memory slot to avoid previous inputs
        memory.insertData("Dialog/LastInput", "")  

        start_time = time.time()
        print("listening...")
        tts.say("Listening")
        while time.time() - start_time < listen_duration:  # Listen for the specified duration
            recognized_text = memory.getData("Dialog/LastInput")
            if recognized_text:
                print("NAO Heard:", recognized_text)
                return recognized_text

        return None

    except Exception as e:
        print("Error:", e)
        return None



review = get_review_text_from_robot(listen_duration=20)

# Call the predict_sentiment function
sentiment_labels = predict_sentiment(str(review))
print(sentiment_labels)
# say the sentiment label
tts.say(str(sentiment_labels))