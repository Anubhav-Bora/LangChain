from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm=GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
prompt = PromptTemplate(
    template='Give me a detailed explanation of the following concept in simple terms: {concept}',
    input_variables=['concept']
)
chain = LLMChain(llm=llm, prompt=prompt)

topic =input("Enter a concept you want to learn about: ")

output=chain.run(topic)
print("output:", output)