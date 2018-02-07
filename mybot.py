import os
import time
import re
from slackclient import SlackClient
import pickle
import re
import numpy as np
from numpy import dot
from numpy.linalg import norm
import datetime
from keras.models import load_model
from predict import decode_sequence
import string

file = open('auth.txt','r') 
token = file.read() 
file.close()

# instantiate Slack client
slack_client = SlackClient(token)
# starterbot's user ID in Slack: value is assigned after the bot starts up
starterbot_id = None
# constants
RTM_READ_DELAY = 0.5 # 1 second delay between reading from RTM


# Load seq2seq pars
encoder_model = load_model('encoder_model.h5')
decoder_model = load_model('decoder_model.h5')
model_hpars = pickle.load( open( "model_hpars.p", "rb" ) )

max_words = model_hpars['max_words']
max_len = model_hpars['max_len']
reverse_input_char_index = model_hpars['reverse_input_char_index']
tokenizer = model_hpars['tokenizer']







MENTION_REGEX = "^<@(|[WU].+)>(.*)"


def clean_command(command):
    ''' To remove punctuations'''
    import string
    non_punc = lambda x:x not in string.punctuation
    command = filter(non_punc, command)
    command = ''.join(list(command))
    return command.lower()


def parse_bot_commands(slack_events):
    """
        Parses a list of events coming from the Slack RTM API to find bot commands.
        If a bot command is found, this function returns a tuple of command and channel.
        If its not found, then this function returns None, None.
    """
    for event in slack_events:
        if event["type"] == "message" and not "subtype" in event:
            user_id, message = parse_direct_mention(event["text"])
            if user_id == starterbot_id:
                return message, event["channel"]
    return None, None

def parse_direct_mention(message_text):
    """
        Finds a direct mention (a mention that is at the beginning) in message text
        and returns the user ID which was mentioned. If there is no direct mention, returns None
    """
    matches = re.search(MENTION_REGEX, message_text)
    # the first group contains the username, the second group contains the remaining message
    return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

def handle_command(command, channel):
    """
        Executes bot command if the command is known
    """
    # Default response is help text for the user
    command = clean_command(command)
    response = None
    # This is where you start to implement more commands!
    global counter;

    
    # Seq2seq prediction
    try:
        query = tokenizer.texts_to_sequences([command])
        query2 = np.zeros(
            (1, max_len, max_words),
            dtype='float32')
        for t, char in enumerate(query[0]):
            query2[:, t, char] = 1. 
        response = decode_sequence(
            input_seq = query2,
            encoder_model = encoder_model,
            decoder_model = decoder_model,
            max_words = max_words,
            max_len = max_len,
            reverse_input_char_index = reverse_input_char_index
            )
        chat_history.append(command)
        chat_history.append(response)


    except:
        response = "I couldn't understand it at this moment, keep talking to me so I can learn from you!"




    if counter%5==0:
        file = open('chat_history.txt','w')
        for chat in chat_history:
            file.write(chat+'\n')
        file.close()

        print('saved!')
        response = response +'\n'+"Oh..Thank you for staying and teach me to talk like human~btw"
        




    

    # Sends the response back to the channel

    slack_client.api_call(
        "chat.postMessage",
        channel=channel,
        text= response
    )

    # slack_client.api_call(
    #     "chat.postMessage",
    #     channel=channel,
    #     text= "\n"+"\n"+"(Your input history:"+"\n"+ "\n".join(chat_history)+")"
    # )





if __name__ == "__main__":
    if slack_client.rtm_connect(with_team_state=False):
        print("Starter Bot connected and running!")
        # Read bot's user ID by calling Web API method `auth.test`
        starterbot_id = slack_client.api_call("auth.test")["user_id"]

        chat_history = []
        counter = 0
        while True:

            command, channel = parse_bot_commands(slack_client.rtm_read())
            if command:
                counter += 1
                handle_command(command, channel)
                

            time.sleep(RTM_READ_DELAY)

    else:
        print("Connection failed. Exception traceback printed above.")