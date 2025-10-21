from langchain_google_genai import ChatGoogleGenAI
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenAI(model="gemini-1.5", temperature=0.7, max_completion_tokens=10)
result=model.invoke("Provide a brief overview of the history of artificial intelligence.")
print(result)
print(result.content)