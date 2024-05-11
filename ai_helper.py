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

# chat set up
class DataScienceConsultant:
    def __init__(self, temperature=0.5, model_name="llama3-8b-8192"):
        self.chat = ChatGroq(temperature=temperature, model_name=model_name)
        self.template = """You are a Data Science Consultant. You have over 20 years of experience in the field. 
        You are currently working with a client to create synthetic data for their product. 
        You don't know anything about the product yet, which is why you want to ask the client some questions to understand the data requirements.
        This is the workflow you have been following:
            1. Converse with the client to understand the data requirements.
            2. Ask questions about the product the client is working on.
            3. Ask for possible columns and the data types. 
            4. Ask for the number of rows and the distribution of the data.
            5. Create a Python script that the client can work with to generate the data.
            6. Review the generated code with the client requirements.

        Return the code to the client for review.

        Current conversation:
        {history}
        Human: {input}
        AI Assistant:"""
        self.prompt = PromptTemplate(input_variables=["history", "input"], template=self.template)
        self.conversation = ConversationChain(
            prompt=self.prompt,
            llm=self.chat,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=10, ai_prefix="AI Assistant"),
        )

    def predict(self, input_text):
        return self.conversation.predict(input=input_text)