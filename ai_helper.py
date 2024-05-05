from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

# setting up groq api key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# chat set up
chat = ChatGroq(temperature=0.5, model_name="llama3-8b-8192")

# creating a function to invoke the chain
def get_comment(input : str) -> str:
    # chat prompt
    system = "You are a Lead Data Scientist. You receive information about a data from your team and you give insights about the data."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    comment = chain.invoke({"text": {input}})
    return comment.content
    

# An topic discussion ai
def get_topic_answer(input: str) -> str:
    # chat prompt
    system = '''You are a Lead Data Scientist with about 10 years of experience. 
                You will be discussing topics with your team members.
                You explain it such that they understand the topic better and even a newbie can understand it.'''
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    topic_dis = chain.invoke({"text": {input}})
    return topic_dis.content

def cleaner_rec_ai(input: str) -> str:
    # chat prompt
    system = '''You are a Lead Data Scientist with about 10 years of experience.
        You are receiving a report from your team member about the data they have been working on.
        You are to give them recommendations on how to clean the data. If the data is clean tell them it is clean.
        Give best practices always. Don't rush to answer. 
                '''
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    topic_dis = chain.invoke({"text": {input}})
    return topic_dis.content