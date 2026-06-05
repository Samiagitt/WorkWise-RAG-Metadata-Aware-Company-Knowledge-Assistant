from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel,Field
from typing import Literal
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser



load_dotenv()
urls = [
    "https://handbook.gitlab.com/handbook/",
    "https://handbook.gitlab.com/handbook/values/",
    "https://handbook.gitlab.com/handbook/communication/",
    "https://handbook.gitlab.com/handbook/company/culture/all-remote/",
    "https://handbook.gitlab.com/handbook/people-group/",
    "https://handbook.gitlab.com/handbook/engineering/",
    "https://handbook.gitlab.com/handbook/security/",
    "https://handbook.gitlab.com/handbook/support/",
    "https://handbook.gitlab.com/handbook/product/"
]

class Metadata(BaseModel):
        department: Literal[
        "company",
        "people_group",
        "engineering",
        "security",
        "support",
        "product",
        "remote_work",
        "communication",
        "general"
    ] = Field(description="Best department for this document")

llm=ChatOpenAI(model="gpt-4o-mini",temperature=0.3)
structured_llm=llm.with_structured_output(Metadata)

metadata_prompt=ChatPromptTemplate.from_template(
        """
Read this document preview and choose the best department.

Allowed departments:
company, people_group, engineering, security, support, product, remote_work, communication, general

Document URL:
{url}

Document preview:
{preview}
"""
)

chain=metadata_prompt|structured_llm

all_docs = []

for url in urls:
    loader = WebBaseLoader(url)
    loaded_docs = loader.load()

    preview = loaded_docs[0].page_content[:2500]

    metadata = chain.invoke({
        "url": url,
        "preview": preview
    })


    for doc in loaded_docs:
        doc.metadata["company"] = "GitLab"
        doc.metadata["source_url"] = url
        doc.metadata["department"] = metadata.department

    all_docs.extend(loaded_docs)

print("\nTotal docs loaded:", len(all_docs))
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_docs)

print("Total chunks:", len(chunks))

embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
persist_directory="Chroma_gitlabDB"

vectorstore=Chroma(
        collection_name="company_db",
        embedding_function=embeddings,
        persist_directory=persist_directory
)

import uuid
ids=[]
for i,chunk in enumerate(chunks):
    chunk_id=str(uuid.uuid4())
    chunk.metadata["chunk_id"]=chunk_id
    chunk.metadata["chunk_number"]=i
    ids.append(chunk_id)
vectorstore.add_documents(
    documents=chunks,
    ids=ids
)
print("Ingestion done")