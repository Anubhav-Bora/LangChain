from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model=GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
result=model.invoke("Provide a brief overview of the history of artificial intelligence.")
print(result)
print(result.content)