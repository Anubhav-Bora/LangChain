from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

model=ChatAnthropic(model="claude-2", temperature=0.7, max_completion_tokens=10)
result=model.invoke("Summarize the plot of Inception in a few sentences.")
print(result)
print(result.content)
