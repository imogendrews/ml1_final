import os
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# -----------------------------
# CONFIG
# -----------------------------
OPENAI_API_KEY = ""  # Replace with your key
DATA_FOLDER = "data"  # Folder where your JSON files are
NUM_FILES = 20  # Adjust if you have more or fewer

# Set API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------------
# 1. Load and Convert JSONs into LangChain Documents
# -----------------------------
all_docs = []

for i in range(1, NUM_FILES + 1):
    file_path = os.path.join(DATA_FOLDER, f"file_{i}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = json.dumps(data, indent=2)
    doc = Document(page_content=text, metadata={"source": f"file_{i}.json"})
    all_docs.append(doc)

print(f"✅ Loaded {len(all_docs)} documents.")

# -----------------------------
# 2. Split Documents into Chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)

print(f"✅ Split into {len(chunks)} chunks.")

# -----------------------------
# 3. Create Vector Store with Embeddings
# -----------------------------
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)

print("✅ Vector store created.")

# -----------------------------
# 4. Create QA Chain
# -----------------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

print("✅ QA chain is ready.")

# -----------------------------
# 5. Ask a Question
# -----------------------------
while True:
    user_query = input("\nAsk a question (or type 'exit'): ")
    if user_query.lower() == "exit":
        break

    result = qa_chain({"query": user_query})
    print("\n🧠 Answer:", result["result"])
    
    print("\n📁 Sources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata["source"])
