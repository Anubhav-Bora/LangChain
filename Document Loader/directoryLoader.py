from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf', #all pdf files in the directory
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load() #important to use lazy_load() to save memory

for document in docs:
    print(document.metadata)