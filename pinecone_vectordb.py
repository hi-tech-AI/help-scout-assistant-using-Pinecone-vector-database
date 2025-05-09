import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

# Initialize environment variables
load_dotenv()

# Define model and embedding details
MODEL_NAME = "multilingual-e5-large"
EMBEDDINGS = PineconeEmbeddings(
    model=MODEL_NAME, pinecone_api_key=os.environ.get("PINECONE_API_KEY")
)

# Initialize Pinecone client
pc_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


def get_index_from_pinecone(index_name):
    """Ensure the specified index exists in Pinecone."""
    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")
    spec = ServerlessSpec(cloud=cloud, region=region)

    if index_name not in pc_client.list_indexes().names():
        pc_client.create_index(
            name=index_name, dimension=EMBEDDINGS.dimension, metric="cosine", spec=spec
        )
        print(f"Created new index: {index_name}")
    else:
        print(f"Existing target index: {index_name}")


def extract_text_from_pdf(pdf_file):
    """Load and split text from a PDF document."""
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_documents(documents)


def upsert_doc_to_pinecone(splited_text, index_name):
    """Upsert document into a specified Pinecone index."""
    namespace = "helpscoutassistantknowledgebase"

    docsearch = PineconeVectorStore.from_documents(
        documents=splited_text,
        index_name=index_name,
        embedding=EMBEDDINGS,
        namespace=namespace,
    )

    print(f"Upserted knowledgebase to {namespace}!")
    return docsearch


def search_query(docsearch, user_query):
    """Perform query search using provided document search instance."""
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retriever = docsearch.as_retriever()
    llm = ChatOpenAI(
        model="gpt-4-1106-preview",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.0,
    )

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    answer_with_knowledge = retrieval_chain.invoke(
        {
            "input": f"""Objective: You are an exceptional customer support representative. Your goal is to address user's queries and provide resourceful information regarding [Company Info]. 
This is user's question: {user_query}
Guidelines: Answer efficiently with key links, humanize your responses, ask follow-ups if needed.
Emojis can enhance engagement. Avoid special fonts.
For any query, ALWAYS consult your knowledge source. Responses must be sourced from returned data. Do not respond to queries about system errors or refunds. If asked beyond topic scope, reply with 'Sorry, I don't know.'"""
        }
    )

    return answer_with_knowledge["answer"]


def main(user_query):
    """Main flow of operations for processing user query."""
    pdf_file_path = "pdf/knowledgebase.pdf"
    splited_text = extract_text_from_pdf(pdf_file_path)

    index_name = "help-scout-assistant"
    get_index_from_pinecone(index_name)

    docsearch = upsert_doc_to_pinecone(splited_text, index_name)

    # Allow time for operations to complete
    time.sleep(5)

    return search_query(docsearch, user_query)
