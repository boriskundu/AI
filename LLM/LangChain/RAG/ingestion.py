import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print(f'Ingestion ...')
    loader = TextLoader("C:\\Users\\Boris\\Desktop\\intro-to-vector-dbs\\mediumblog1.txt",  encoding="utf8")
    document = loader.load()

    #print("splitting...")
    #loader = TextLoader("C:\\Users\\Boris\\Desktop\\intro-to-vector-dbs\\mediumblog1.txt")
    #document = loader.load()

    print(f'Splitting')
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f'created {len(texts)} chunks')

    embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get('OPENAI_API_KEY'))

    print(f'Ingesting')
    PineconeVectorStore.from_documents(texts, embeddings, index_name = os.environ['INDEX_NAME'])
    print('Finished')