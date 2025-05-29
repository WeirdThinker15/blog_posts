import os 
import pandas as pd
# from IPython.display import Markdown, HTML, display

from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

from helper import get_openai_api_key

# Create instance of OpenAPI ChatGPT 
llm = ChatOpenAI(
    api_key=get_openai_api_key()
)

# Create a message to send to ChatGPT 
message = HumanMessage(
    content="Translate this sentence from English "
    "to French and Spanish. I like red cars and "
    "blue houses, but my dog is yellow."
)

output = llm.invoke([message])
print(output.text())

