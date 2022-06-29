from email import message
from enum import Enum
from typing import List

from fastapi import status
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
from loguru import logger
from downloadmodel import download_and_save_chatbot_model

import nltk
import numpy as np
import keras
from nltk.stem.lancaster import LancasterStemmer
import spacy

router = APIRouter()

class StatusEnum(str, Enum):
    OK = "OK"
    FAILURE = "FAILURE"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class HealthCheck(BaseModel):
    title: str = Field(..., description="API title")
    description: str = Field(..., description="Brief description of the API")
    version: str = Field(..., description="API server version number")
    status: StatusEnum = Field(..., description="API current status")

class userInput(BaseModel):
    message: str

class modelOutput(BaseModel):
    tag: str
    recognation: List

data = {}

@router.get(
    "/status",
    response_model=HealthCheck,
    status_code=status.HTTP_200_OK,
    tags=["Health Check"],
    summary="Performs health check",
    description="Performs health check and returns information about running service.",
)
def health_check():
    return {
        "title": "ChatBot Service",
        "description": "This is a test desc",
        "version": "0.0.0",
        "status": StatusEnum.OK,
    }

@router.on_event("startup")
async def downloading_offensive_model() -> bool:
    logger.info("Downloading Chatbot Model ........")
    try:

        result = await download_and_save_chatbot_model()
        if result:
            logger.info("Model Has Been Downloaded and Saved ...")
            return True
    except Exception as e:
        logger.exception(e.__cause__)


@router.on_event("startup")
def load_model():
    logger.info("Loading Model to Memory ....")
    data[0] = keras.models.load_model('model.h5')
    data[1] = spacy.load('visit_egypt')
    logger.info("Model has been loaded ...")
    return data


words =  ["'s", '50', 'a', 'about', 'am', 'anyon', 'ar', 'artifact', 'be', 'bye', 'chang', 'clin', 'convert', 'dang', 'day', 'do', 'doll', 'eat', 'emerg', 'euro', 'find', 'forecast', 'going', 'good', 'goodby', 'hello', 'help', 'hi', 'hotel', 'how', 'i', 'in', 'insight', 'is', 'it', 'know', 'lat', 'lik', 'loc', 'me', 'medicin', 'nee', 'next', 'now', 'of', 'pol', 'pound', 'rain', 'resta', 'right', 'see', 'sleep', 'sup', 'tel', 'temp', 'thank', 'that', 'the', 'ther', 'to', 'tomorrow', 'top', 'want', 'weath', 'what', 'you']
labels =  {0:'Clinic',1: 'Hotel',2: 'Restaurant',3: 'conversation',4: 'currency',5: 'goodbye',6: 'greeting',7: 'info',8:'police',9:'thanks',10:'weather'}
nltk.download('punkt')
stemmer =  LancasterStemmer()

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)



def net(word) :
     model_entity = data[1]
     sentence = []
     for ent in model_entity(word).ents:
        sentence.append({"Name": ent.text , "Label": ent.label_})
     return sentence


@router.post('/predict', response_model = modelOutput)
def chat(inp: userInput):
        model_intity = data[0]
        wordbag =  bag_of_words(inp.message, words)
        recognation =  net(inp.message) 
        class_intent = model_intity.predict(np.array([wordbag]))
        class_index = np.argmax(class_intent)
        tag = labels[class_index]
        return modelOutput(tag=tag, recognation=recognation)