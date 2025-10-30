from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

promt1=PromptTemplate(
    template="Write a joke about {topic}.",
    input_variables=["topic"],
)

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

parser=StrOutputParser()

promt2=PromptTemplate(
    template="Explain the following joke : {joke}",
    input_variables=["joke"],
)

chain=RunnableSequence(promt1,model,parser,promt2,model,parser)

topic=input("Enter a topic for a joke: ")

print(chain.invoke({"topic":topic}))