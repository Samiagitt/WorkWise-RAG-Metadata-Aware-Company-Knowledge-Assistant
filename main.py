from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class QueryRoute(BaseModel):
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
    ] = Field(description="Best department/category for the user's question")


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = Chroma(
    collection_name="company_db",
    embedding_function=embeddings,
    persist_directory="Chroma_gitlabDB"
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

router_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

structured_router_llm = router_llm.with_structured_output(QueryRoute)

router_prompt = ChatPromptTemplate.from_template(
    """
You are a router for a company knowledge assistant.

Your job is to choose the best department for the user's question.

Allowed departments:
- company
- people_group
- engineering
- security
- support
- product
- remote_work
- communication
- general

Rules:
- values, culture, mission, company information -> company
- HR, people, benefits, hiring, team member lifecycle -> people_group
- code, development, engineering, code review -> engineering
- security, compliance, privacy, data protection -> security
- customer support, tickets, customer help -> support
- product strategy, product principles, product management -> product
- remote work, work from home, all-remote -> remote_work
- communication style, meetings, written communication -> communication
- unclear or broad question -> general

Question:
{question}
"""
)

router_chain = router_prompt | structured_router_llm


def retrieve_docs(question):
    route = router_chain.invoke(
        {
            "question": question
        }
    )

    search_kwargs = {
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }

    if route.department != "general":
        search_kwargs["filter"] = {
            "department": route.department
        }

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    retrieved_docs = retriever.invoke(question)

    if len(retrieved_docs) == 0:
        print("\nNo filtered documents found. Running fallback retrieval without filter...")

        fallback_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.5
            }
        )

        retrieved_docs = fallback_retriever.invoke(question)

    return retrieved_docs, route.department

question = "How does GitLab handle remote work?"

retrieved_docs, selected_department = retrieve_docs(question)


print("\n===== RETRIEVED SOURCES =====")

for i, doc in enumerate(retrieved_docs, start=1):
    print(f"\nDOC {i}")
    print("Department:", doc.metadata.get("department"))
    print("Source URL:", doc.metadata.get("source_url"))
    print("Preview:", doc.page_content[:300])
    print("-" * 80)



context = "\n\n".join(
    [doc.page_content for doc in retrieved_docs]
)

answer_prompt = ChatPromptTemplate.from_template(
    """
You are WorkWise RAG, a professional company knowledge assistant.

Answer the user's question using ONLY the retrieved GitLab handbook context.

Rules:
- Be clear and concise.
- Use simple language.
- Use bullet points if helpful.
- Do not make up company policies.
- If the answer is not in the context, say:
  "I could not find this information in the stored company documents."

Retrieved Context:
{context}

User Question:
{question}

Answer:
"""
)

rag_chain = answer_prompt | llm | StrOutputParser()

answer = rag_chain.invoke(
    {
        "context": context,
        "question": question
    }
)

print("\n===== FINAL ANSWER =====")
print(answer)