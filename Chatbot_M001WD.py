import os 
from google import genai
#import warnings
#warnings.filterwarnings("ignore")
client = genai.Client(api_key="AQ.Ab8RN6Jz6v4gEYA5JAHUvyjjQHPA_xqOpwx_qZCuSdhfKSs8Xg")
import random
import json
import torch
from model import WD
from chatbot_01 import bag_of_words, tokenize
#import logging
#logging.getLogger("google").setLevel(logging.ERROR)
#logging.getLogger("google.genai").setLevel(logging.ERROR)
import io
from contextlib import redirect_stderr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('commands.json', 'r') as j:
    commands = json.load(j)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = WD(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#bot_name = "WD World MODEL 001"
def get_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim = 1)
    tag = tags[predicted.item()] 

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for command in commands['commands']:
            if tag == command['tag']:
                return random.choice(command['responses'])
    else:
        try:
            user_query = " ".join(sentence)
            f = io.StringIO()
            with redirect_stderr(f):
                response = client.models.generate_content(
                    model = "gemini-3.6-flash",
                    contents = user_query
                )
            return response.text
        except Exception:
            return "I am unable to answer right now."
 