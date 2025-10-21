import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from config import S3_BUCKET_NAME, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
import os


class DocumentLoader:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def load_documents_from_s3(self):
        """Load all documents from S3 bucket"""
        documents = []

        # List and load test_set files
        response = self.s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix='test_set/')

        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.txt'):
                print(f" Loading {key}...")

                # Get file content from S3
                obj_data = self.s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
                content = obj_data['Body'].read().decode('utf-8')

                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={"source": key, "department": key.split('/')[-1].replace('.txt', '')}
                )
                documents.append(doc)

        print(f" Loaded {len(documents)} documents from S3")
        return documents

    def chunk_documents(self, documents):
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        print(f" Split into {len(chunks)} chunks")
        return chunks

    def create_vector_store(self, chunks):
        """Create FAISS vector store from chunks"""
        print(" Creating vector store...")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local("vector_store")
        print(" Vector store created and saved locally")
        return vector_store

    def load_training_data(self):
        """Load QnA training data from S3"""
        try:
            obj_data = self.s3.get_object(
                Bucket=S3_BUCKET_NAME,
                Key='training_set/qna_training_data.csv'
            )
            content = obj_data['Body'].read().decode('utf-8')

            # Save locally for easy access
            with open('training_data.csv', 'w', encoding='utf-8') as f:
                f.write(content)

            print(" Training data loaded from S3")
            return 'training_data.csv'
        except Exception as e:
            print(f" Error loading training data: {e}")
            return None


if __name__ == "__main__":
    loader = DocumentLoader()

    # Step 1: Load documents from S3
    documents = loader.load_documents_from_s3()

    # Step 2: Chunk documents
    chunks = loader.chunk_documents(documents)

    # Step 3: Create vector store
    vector_store = loader.create_vector_store(chunks)

    # Step 4: Load training data
    training_file = loader.load_training_data()

    print(" Document ingestion completed!")