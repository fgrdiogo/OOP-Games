import os
import json
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


os.environ["OPENAI_API_KEY"] = "SUA_CHAVE_AQUI"  # Substitua pela sua chave


with open("referred_legal_documents_QA_2024_v1.0.json", "r", encoding="utf-8") as f:
    data = json.load(f)


texts_raw = []
for doc in data:
    filename = doc.get("filename", "sem_nome.txt")
    filedata = doc.get("filedata", "")
    texts_raw.append(f"{filename}\n{filedata}")

# Concatenar tudo em um único texto
all_text = "\n\n".join(texts_raw)


text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  
    chunk_size=1000,              
    chunk_overlap=200             
)

texts = text_splitter.split_text(all_text)
print(f"✅ Total de chunks gerados: {len(texts)}")


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


vectorstore = FAISS.from_texts(texts, embedding=embeddings)
vectorstore.save_local("faiss_index_referred_docs")

print("indice do FAISS salvo")


query = input("Digite uma consulta para buscar no índice: ")
results = vectorstore.similarity_search(query, k=3)

for i, r in enumerate(results, 1):
    print(f"\n🔹 Resultado {i}:")
    print(r.page_content[:400], "...")
