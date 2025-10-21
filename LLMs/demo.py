from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm=OpenAI(model="gpt-3.5-turbo")

result=llm.invoke("Tell me a joke")
print(result)