# chat-raceita-experimentos

1. Indexar os documentos do arquivo referred_legal_documents_QA_2024_v1.0.jso

split:

from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=200
)
texts = text_splitter.split_text(document)

Documento a ser indexado tem que ser:
{
page_content=filedata,
metadata={"filename": filename}
}

Usar o modelo text-embedding-3-small
Base: https://python.langchain.com/docs/integrations/vectorstores/faiss/
https://python.langchain.com/docs/concepts/text_splitters/#length-based
Salvar index no disco


Carregando o dataset:
https://huggingface.co/datasets/unicamp-dl/BR-TaxQA-R/tree/main
