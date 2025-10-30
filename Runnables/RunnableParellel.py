from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

parser=StrOutputParser()
promt1=PromptTemplate(
    template="Write a joke about {topic_joke}.",
    input_variables=["topic_joke"],
)
promt2=PromptTemplate(
    template="Write a fun fact about {topic_fun_fact}.",
    input_variables=["topic_fun_fact"],
)

chain = RunnableParallel({
   "joke": RunnableSequence(promt1, model, parser),
   "fun_fact": RunnableSequence(promt2, model, parser),
})

topic_joke=input("Enter a topic for a joke ")
topic_fun_fact=input("Enter a topic for a fun fact ")

print(chain.invoke({"topic_joke": topic_joke, "topic_fun_fact": topic_fun_fact}))
