# document_loader.py - FIXED VERSION
import boto3
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import S3_BUCKET_NAME, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
import time


class DocumentLoader:
    def __init__(self):
        self.s3 = boto3.client('s3')
        # Use HuggingFace embeddings instead of OpenAI to avoid API costs
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_documents_from_s3(self):
        """Load all documents from S3 bucket"""
        print("Loading documents from S3...")
        documents = []

        try:
            # List and load test_set files
            response = self.s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix='test_set/')

            if 'Contents' not in response:
                print("No files found in S3 bucket")
                return documents

            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.txt'):
                    print(f"Loading {key}...")

                    # Get file content from S3
                    obj_data = self.s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
                    content = obj_data['Body'].read().decode('utf-8')

                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": key,
                            "department": key.split('/')[-1].replace('.txt', ''),
                            "file_size": len(content)
                        }
                    )
                    documents.append(doc)

            print(f"✓ Loaded {len(documents)} documents from S3")
            return documents

        except Exception as e:
            print(f"✗ Error loading from S3: {e}")
            return documents

    def chunk_documents(self, documents):
        """Split documents into chunks"""
        if not documents:
            print("No documents to chunk")
            return []

        print("Chunking documents...")
        try:
            chunks = self.text_splitter.split_documents(documents)
            print(f"✓ Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"✗ Error chunking documents: {e}")
            return []

    def create_vector_store(self, chunks):
        """Create FAISS vector store from chunks"""
        if not chunks:
            print("No chunks available for vector store")
            return None

        print("Creating vector store...")
        try:
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local("vector_store")
            print("✓ Vector store created and saved locally")
            return vector_store
        except Exception as e:
            print(f"✗ Error creating vector store: {e}")
            return None

    def load_training_data(self):
        """Load QnA training data from S3"""
        print("Loading training data...")
        try:
            obj_data = self.s3.get_object(
                Bucket=S3_BUCKET_NAME,
                Key='training_set/qna_training_data.csv'
            )
            content = obj_data['Body'].read().decode('utf-8')

            # Save locally for easy access
            os.makedirs('data', exist_ok=True)
            with open('data/training_data.csv', 'w', encoding='utf-8') as f:
                f.write(content)

            print("✓ Training data loaded from S3")
            return 'data/training_data.csv'
        except Exception as e:
            print(f"✗ Error loading training data: {e}")
            return None

    def test_s3_connection(self):
        """Test S3 connection and list files"""
        print("Testing S3 connection...")
        try:
            response = self.s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix='')
            if 'Contents' in response:
                print("✓ S3 Connection successful. Files found:")
                for obj in response['Contents']:
                    print(f"  - {obj['Key']}")
            else:
                print("⚠ S3 connected but no files found")
            return True
        except Exception as e:
            print(f"✗ S3 Connection failed: {e}")
            return False


def main():
    print("Starting Document Ingestion Pipeline")
    print("=" * 50)

    loader = DocumentLoader()

    # Test S3 connection first
    if not loader.test_s3_connection():
        print("S3 connection failed. Please check your AWS credentials and bucket name.")
        return

    # Step 1: Load documents from S3
    documents = loader.load_documents_from_s3()

    if not documents:
        print("No documents found in S3. Please check if files are uploaded correctly.")
        return

    # Step 2: Chunk documents
    chunks = loader.chunk_documents(documents)

    if not chunks:
        print("Document chunking failed.")
        return

    # Step 3: Create vector store
    vector_store = loader.create_vector_store(chunks)

    if not vector_store:
        print("Vector store creation failed.")
        return

    # Step 4: Load training data
    training_file = loader.load_training_data()

    print("\n" + "=" * 50)
    print(" Document ingestion completed!")
    print("✓ Vector store ready for RAG pipeline")
    print("✓ Training data loaded")
    print("✓ Ready for model evaluation!")
    print("=" * 50)


if __name__ == "__main__":
    main()