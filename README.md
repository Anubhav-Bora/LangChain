# LangChain Learning Guide

A comprehensive guide to LangChain fundamentals with practical examples.

## Table of Contents
- [What is LangChain?](#what-is-langchain)
- [LLMs vs Chat Models](#llms-vs-chat-models)
- [Prompts](#prompts)
- [Output Parsers](#output-parsers)
- [Chains](#chains)
- [Document Loaders](#document-loaders)
- [Text Splitters](#text-splitters)
- [Embeddings](#embeddings)
- [Vector Stores](#vector-stores)
- [Retrievers](#retrievers)
- [Runnables](#runnables)
- [Structured Output](#structured-output)

---

## What is LangChain?

**LangChain** is a framework for building applications powered by Large Language Models (LLMs). It helps you:
- Connect LLMs with external data sources
- Chain multiple operations together
- Build context-aware AI applications
- Create RAG (Retrieval Augmented Generation) systems

---

## LLMs vs Chat Models

### LLMs
Traditional text completion models that take a string input and return a string output.

```python
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo")
result = llm.invoke("Tell me a joke")
print(result)
```

### Chat Models
Models designed for conversations, taking messages as input and returning messages.

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=10)
result = model.invoke("Explain the theory of relativity in simple terms.")
print(result.content)
```

**Key Difference**: Chat models maintain conversation context and use message formats (system, human, AI).

---

## Prompts

Prompts are templates that format inputs for LLMs.

### ChatPromptTemplate
```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'Dusra'})
```

### PromptTemplate
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)
```

**Purpose**: Reusable, dynamic text templates with variable placeholders.

---

## Output Parsers

Parse and structure LLM outputs into desired formats.

### String Output Parser
```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({'topic': 'cricket'})
```

### Pydantic Output Parser
```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    city: str = Field(description="The city where the person lives")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
```

**Purpose**: Convert raw text into structured data (strings, JSON, Python objects).

---

## Chains

Chains connect multiple components (prompts, models, parsers) into a pipeline.

### Simple Chain
```python
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic': 'cricket'})
```

### Sequential Chain
Execute multiple steps in sequence, passing output from one to the next.

```python
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic': 'Unemployment in India'})
```

**Purpose**: Automate multi-step workflows where each step depends on the previous one.

---

## Document Loaders

Load documents from various sources into LangChain.

### Text Loader
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader('cricket.txt', encoding='utf-8')
docs = loader.load()  # Returns list of Document objects

# Access content and metadata
print(docs[0].page_content)  # Text content
print(docs[0].metadata)      # File metadata
```

### Other Loaders
- **CSV Loader**: Load CSV files
- **PyPDF Loader**: Extract text from PDFs
- **WebBase Loader**: Scrape web pages
- **Directory Loader**: Load all files from a directory

**Purpose**: Import data from different sources for processing.

---

## Text Splitters

Split large documents into smaller chunks for processing.

### Semantic Chunker
Splits text based on semantic meaning, not just character count.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

text_splitter = SemanticChunker(
    OpenAIEmbeddings(), 
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

docs = text_splitter.create_documents([sample_text])
```

### Other Splitters
- **Length-Based**: Split by character/token count
- **Markdown Splitter**: Preserve markdown structure
- **Code Splitter**: Language-aware code splitting
- **Recursive Splitter**: Try multiple split strategies

**Purpose**: Break large texts into manageable chunks that fit LLM context windows.

---

## Embeddings

Convert text into numerical vectors (embeddings) for semantic search.

```python
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)
# Returns list of vectors representing semantic meaning
```

**Purpose**: Enable semantic similarity search and document matching.

---

## Vector Stores

Store and search embeddings efficiently.

Vector stores save document embeddings and allow similarity searches:
- **FAISS**: Facebook AI Similarity Search (local, fast)
- **Chroma**: Open-source embedding database
- **Pinecone**: Managed vector database
- **Weaviate**: Open-source vector search engine

```python
# Typical workflow:
# 1. Load documents
# 2. Split into chunks
# 3. Create embeddings
# 4. Store in vector database
# 5. Query for similar documents
```

**Purpose**: Efficiently search through large document collections using semantic similarity.

---

## Retrievers

Retrieve relevant documents based on queries.

Retrievers find the most relevant documents from a vector store:

```python
# Create retriever from vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Retrieve relevant documents
docs = retriever.invoke("What is machine learning?")
```

**Purpose**: Bridge between user queries and relevant document chunks for RAG systems.

---

## Runnables

Advanced execution patterns for complex workflows.

### RunnableParallel
Execute multiple chains simultaneously.

```python
from langchain.schema.runnable import RunnableSequence, RunnableParallel

chain = RunnableParallel({
   "joke": RunnableSequence(promt1, model, parser),
   "fun_fact": RunnableSequence(promt2, model, parser),
})

result = chain.invoke({
    "topic_joke": "cats", 
    "topic_fun_fact": "space"
})
```

### Other Runnables
- **RunnableSequence**: Chain operations sequentially
- **RunnableLambda**: Custom Python functions in chains
- **RunnableBranch**: Conditional execution paths
- **RunnablePassthrough**: Pass data through unchanged

**Purpose**: Build complex, efficient pipelines with parallel execution and conditional logic.

---

## Structured Output

Force LLMs to return data in specific formats.

### with_structured_output (Pydantic)
```python
from pydantic import BaseModel, Field
from typing import Optional, Literal

class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes discussed")
    summary: str = Field(description="Brief summary")
    sentiment: Literal["pos", "neg"] = Field(description="Sentiment")
    pros: Optional[list[str]] = Field(default=None, description="Pros")
    cons: Optional[list[str]] = Field(default=None, description="Cons")
    name: Optional[str] = Field(default=None, description="Reviewer name")

structured_model = model.with_structured_output(Review)
result = structured_model.invoke("Review text here...")
```

### Benefits
- Type-safe outputs
- Automatic validation
- No parsing errors
- Direct Python object access

**Purpose**: Get reliable, structured data from LLMs for use in applications.

---

## Installation

```bash
pip install -r requirements.txt
```

### Key Dependencies
```
langchain
langchain-core
langchain-openai
langchain-anthropic
langchain-google-genai
langchain-huggingface
python-dotenv
pydantic
```

---

## Environment Setup

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

---

## Common Patterns

### RAG (Retrieval Augmented Generation)
1. **Load** documents
2. **Split** into chunks
3. **Embed** chunks
4. **Store** in vector database
5. **Retrieve** relevant chunks for query
6. **Generate** answer with LLM + context

### Basic Chain Pattern
```python
chain = prompt | model | parser
result = chain.invoke(input_data)
```

### Parallel Processing
```python
chain = RunnableParallel({
    "task1": chain1,
    "task2": chain2
})
```

---

## Best Practices

1. **Use environment variables** for API keys (never hardcode)
2. **Set temperature** appropriately (0 for factual, 0.7-1.0 for creative)
3. **Limit tokens** to control costs and response length
4. **Chunk documents** appropriately for your use case
5. **Use structured output** when you need reliable data formats
6. **Test with different models** to find the best fit
7. **Handle errors** gracefully with try-except blocks

---

## Project Structure

```
├── LLMs/                    # Basic LLM usage
├── ChatModels/              # Chat model implementations
├── Prompts/                 # Prompt templates
├── chains/                  # Chain examples
├── outPutParser/            # Output parsing strategies
├── Document Loader/         # Document loading utilities
├── text_splitter/           # Text splitting techniques
├── EmbeddedModels/          # Embedding examples
├── VectorStore/             # Vector database integration
├── Retriever/               # Retrieval patterns
├── Runnables/               # Advanced runnable patterns
├── Structured_Output/       # Structured output examples
└── requirements.txt         # Dependencies
```

---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [OpenAI API](https://platform.openai.com/)
- [Google Gemini](https://ai.google.dev/)
- [Hugging Face](https://huggingface.co/)

---

## Summary

LangChain simplifies building LLM applications by providing:
- **Modular components** that work together
- **Standardized interfaces** across different LLM providers
- **Pre-built utilities** for common tasks
- **Flexible chains** for complex workflows
- **RAG capabilities** for knowledge-based applications

Start with simple chains, then progressively add document loading, embeddings, and retrieval as needed!
