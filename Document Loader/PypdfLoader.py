from langchain_community.document_loaders import PyPDFLoader ## used only have text pdf not photo pdf or something there are more pdf loader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
parser=StrOutputParser()

loader=PyPDFLoader('B.TECH1_.pdf')
docs=loader.load() # we can use lazy_load() for large files
##print(len(docs)) #number of pages in the pdf file (docs)

#there are pdf loaders for different purposes see the notes(pdf)