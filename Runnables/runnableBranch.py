from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableBranch, RunnablePassthrough

load_dotenv()

prompt1=PromptTemplate(
    template="Write a report about {topic}.",
    input_variables=["topic"],
)
promt2=PromptTemplate(
    template="Summarize the following report : {report}",
    input_variables=["report"],
)
model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
parser=StrOutputParser()

report_gen_chain=RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 50,
     RunnableSequence(promt2, model, parser)),
    RunnablePassthrough()
)

final_chain=RunnableSequence(report_gen_chain, branch_chain)
topic=input("Enter a topic for a report: ")
print(final_chain.invoke({"topic":topic}))
    