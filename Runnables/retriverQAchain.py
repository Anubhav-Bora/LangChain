from langchain.document_loaders import TextLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

loader=TextLoader('sample.txt') #not present now
documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs=text_splitter.split_documents(documents)


vectorstore=FAISS.from_documents(docs, GoogleGenerativeAIEmbeddings())

retriever=vectorstore.as_retriever()

llm=GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)


qa_chain=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


result=qa_chain.run("what is the history of ai")
print(result)
