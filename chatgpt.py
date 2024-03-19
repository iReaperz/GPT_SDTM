import os
import sys

import openai
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

with open('API_KEY_SDTM.txt','r') as f:
    os.environ["OPENAI_API_KEY"] = f.read()

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = DirectoryLoader("data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

# chain = ConversationalRetrievalChain.from_llm(
#   llm=ChatOpenAI(model="gpt-3.5-turbo"),
#   retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
# )

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  # result = chain({"question": query, "chat_history": chat_history})
  # print(result['answer'])

  print(index.query(query))
  
  chat_history.append((query, index.query(query)))
  query = None
