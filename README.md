# 🏢 WorkWise RAG

### Metadata-Aware Company Knowledge Assistant using LangChain

WorkWise RAG is a **company knowledge assistant** built with **LangChain, OpenAI Embeddings, ChromaDB, and LLM-based metadata filtering**.

Instead of searching all documents blindly, this project uses an **LLM router** to understand the user’s question and search the most relevant company department, such as HR, engineering, security, support, product, communication, or remote work.

---

## ✨ Project Overview

This project uses GitLab’s public handbook pages as a real-world company knowledge base.

The goal is to simulate how an internal company assistant could help employees quickly find answers from large company documentation.

Example questions:

* 💬 How does GitLab communicate?
* 🌎 How does GitLab handle remote work?
* 🧑‍💼 What does the People Group do?
* 🛡️ What does the Security handbook cover?
* 🛠️ What does the Engineering handbook discuss?
* 📦 What does the Product handbook cover?

---

## 🚀 Key Features

* 🌐 Ingests real company handbook pages from GitLab
* 🧠 Uses LLM-assisted metadata labeling during ingestion
* 🗂️ Stores documents in ChromaDB with department metadata
* 🔀 Uses an LLM router to classify user questions
* 🎯 Applies metadata-filtered retrieval based on department
* 🔁 Includes fallback retrieval if filtered retrieval fails
* 🧾 Shows retrieved source URLs and department metadata
* 💬 Generates grounded answers using retrieved context only

---

## 🧠 How It Works

```text
GitLab Handbook Pages
        ↓
WebBaseLoader
        ↓
LLM Generates Metadata
        ↓
Text Splitting
        ↓
OpenAI Embeddings
        ↓
ChromaDB Vector Store
        ↓
User Question
        ↓
LLM Router Selects Department
        ↓
Metadata-Filtered Retrieval
        ↓
Fallback Retrieval if Needed
        ↓
Final Answer with Sources
```

---

## 🔍 Why Metadata Filtering?

In real companies, internal documents can become huge and messy. Searching the entire knowledge base every time may return irrelevant results.

This project improves retrieval by using metadata such as:

* `company`
* `department`
* `source_url`
* `chunk_id`
* `chunk_number`

For example:

```text
Question: How does GitLab handle remote work?
Router Output: remote_work
Retriever Filter: department = remote_work
```

This helps the assistant search the most relevant part of the knowledge base instead of retrieving unrelated documents.

---

## 🤖 LLM Router

The router LLM does not answer the question.
Its job is to decide which department should be searched.

Example:

```text
User Question: What does the security handbook cover?
Selected Department: security
```

Then ChromaDB retrieves only documents where:

```python
department = "security"
```

This makes the RAG pipeline more focused and closer to real enterprise search systems.

---

## 🔁 Fallback Retrieval

Sometimes metadata filtering can be too strict.
If filtered retrieval does not return documents, the system automatically retries without the metadata filter.

This improves reliability and prevents the assistant from failing too early.

```text
Filtered Retrieval
        ↓
No docs found?
        ↓
Fallback Retrieval without filter
```

---

## 🛠️ Tech Stack

* 🐍 Python
* 🔗 LangChain
* 🧠 OpenAI Embeddings
* 💬 OpenAI Chat Model
* 🗂️ ChromaDB
* 🌐 WebBaseLoader
* ✂️ RecursiveCharacterTextSplitter
* 🎯 MMR Retrieval
* 🔀 LLM Query Router

---

## 📌 What I Learned

Through this project, I practiced:

* building a real-world RAG pipeline
* ingesting web-based company documentation
* generating metadata with an LLM
* storing metadata in ChromaDB
* using metadata filters during retrieval
* designing an LLM-based query router
* adding fallback retrieval
* grounding answers in retrieved context

---

## 🧪 Example

**Question:**

```text
How does GitLab handle remote work?
```

**Router Result:**

```text
Selected department: remote_work
```

**Retrieved Source:**

```text
https://handbook.gitlab.com/handbook/company/culture/all-remote/
```

**Answer:**

```text
The answer is generated only from the retrieved GitLab handbook context.
```

---

## 📈 Why This Project Is Useful

This project represents a common enterprise AI use case:

> Helping employees search large internal knowledge bases more efficiently.

Instead of creating a simple document chatbot, this project adds smarter retrieval using:

* LLM-assisted metadata labeling
* department-based query routing
* metadata-filtered retrieval
* fallback retrieval
* source-grounded answer generation

---

## 🔮 Future Improvements

* 🧠 Add MultiQuery Retriever after stable filtering
* ✅ Add document relevance grading
* 🔍 Add query rewriting
* 📊 Add RAG evaluation with RAGAS
* 🧩 Convert the pipeline into LangGraph nodes
* 🔁 Add CRAG-style corrective retrieval
* 🤖 Add Self-RAG answer checking
* 🌐 Add a Streamlit UI later

---

## ⚠️ Note

This project uses public GitLab handbook pages for educational and portfolio purposes. It is not affiliated with GitLab.

