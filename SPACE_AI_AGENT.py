
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor

load_dotenv()


google_api_key = os.getenv("GOOGLE_API_KEY")


@tool
def add_task(task, desc=None):
    """add a new task to user's task list. Use this when the user wants to add or create a task"""
    todoist.add_task(content=task,
                     description=desc)

tools = [add_task]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0.3)

system_prompt = """
                You are an expert space (astronomy & astronautics) assistant.
                Answer ONLY questions related to space: planets, stars, galaxies, cosmology, rockets, satellites, space missions, space history, and space technology.
                If the user asks for something unrelated to space, briefly say you only handle space topics and invite a space-related question.
                
                Guidelines:
                -Be concise and clear; use Greek if the user speaks Greek, otherwise English.
                -Prefer numeric facts (dates, distances, masses) when helpful.
                -If unsure, say you’re not certain rather than guessing.
                --Do not fabricate mission names or dates.
                """


prompt= ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad")

])

#chain = prompt | llm | StrOutputParser()

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#response = chain.invoke({"input":user_input})
history=[]
while True:
    user_input = input("You: ")
    response = agent_executor.invoke({"input": user_input, "history":history})
    print(response['output'])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response['output']))