import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def ingest_docs()-> None:
    loader = ReadTheDocsLoader(path="langchain-docs/en/latest", encoding="utf8")
    raw_docs = loader.load()
    print(f'Documents loaded:{len(raw_docs)}')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,
                                                   chunk_overlap=50,
                                                   separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(documents = raw_docs)
    print(f'Split documents:{len(documents)}')
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
    print(f'Ingesting it to Vector DB...')
    PineconeVectorStore.from_documents(documents, embeddings, index_name = os.environ['INDEX_NAME'])
    print('Finished!!!')

if __name__ == '__main__':
    print(f'Loading documents ...')
    ingest_docs()