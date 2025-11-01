from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
parser=StrOutputParser()

promt=PromptTemplate(
    template="Summarize the following text : {text}",
    input_variables=["text"],
)

loader=TextLoader('cricket.txt', encoding='utf-8')

docs=loader.load() # we can use lazy_load() for large files

##print(type(docs)) #list

##print(docs[0]) #Document object
##print(docs[0].page_content) #text content of the document
##print(docs[0].metadata) #metadata of the document

chain =promt | model | parser
result=chain.invoke({"text":docs[0].page_content})
print(result)




