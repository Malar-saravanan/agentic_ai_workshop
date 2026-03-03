# The Agentic Future: Reinventing Software Beyond SaaS

**Architecting Autonomy — Building Goal-Driven Agents with Groq & CrewAI**

Hands-on workshop on building multi-agent AI systems. You'll understand how AI agents reason and execute tasks, learn planning, memory, and tool-calling, and design your own goal-driven agent.

**Resource Person:** Ms. Malar Saravanan | Senior AI/ML Engineer, Adobe  
**Event:** Yugam 2026 · GDG KCT · Kumaraguru | 4th March 2026  
**Register:** [yugam.in/w/CSE2](https://yugam.in/w/CSE2)

## 📋 Agenda at a Glance

| Time          | Topic                                                       | What You Build                                                             |
| ------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------- |
| 9:30 – 10:30  | **1. The Agentic Paradigm** — Beyond Chatbots to Operators  | Hello World agent (`01_hello_agent.py`)                                    |
| 10:30 – 11:15 | **2. Think → Act → Observe** — ReAct & Real-Time Reasoning  | Cognitive loops; Groq speed; reasoning vs execution models                 |
| 11:15 – 12:15 | **3. Agents with Superpowers** — Tools, Web Search & RAG   | Researcher agent (`02_tools.py`, `03_researcher.py`)                        |
| 12:15 – 1:15  | **4. Collaborative Intelligence** — Multi-Agent Architecture| 3-agent Shopping Crew (`04_agents.py` → `06_crew.py`)                      |
| 1:15 – 3:30   | **5. Production-Ready Systems** — Routing & Interfaces      | LLM router + CLI + Streamlit (`orchestrator.py`, `cli.py`, `app.py`)      |

## 🛠️ Prerequisites

- Python 3.10+
- [Groq API key](https://console.groq.com) (free)
- [Serper API key](https://serper.dev) (free tier for web search)
- Basic Python knowledge

## 🚀 Setup (5 minutes)

```bash
# 1. Clone or download this folder
cd KCT_AgenticAI

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env and add:
#   GROQ_API_KEY=your_groq_key
#   SERPER_API_KEY=your_serper_key
# Optional: LLM_MODEL_CANDIDATES=groq/llama-3.1-8b-instant,groq/llama-3.3-70b-versatile
```

## ▶️ Running the Labs

```bash
# Phase 1: Hello World
python 01_hello_agent.py

# Phase 2: Researcher with tools
python 03_researcher.py

# Phase 3: Full 3-agent crew
python 06_crew.py "wireless earbuds" "50"

# Phase 4: Orchestrator (CLI)
python cli.py "What is your return policy?"
python cli.py "Find best earbuds under $50"

# Streamlit Demo
streamlit run app.py
```

## 📁 File Guide

| File                 | Phase | What It Does                                  |
| -------------------- | ----- | --------------------------------------------- |
| `01_hello_agent.py`  | 1     | Single agent answers a product question       |
| `02_tools.py`        | 2     | Web Search + RAG tools                        |
| `03_researcher.py`   | 2     | Agent uses both tools to research products    |
| `04_agents.py`       | 3     | Defines 3 agents (2 with tools, 1 pure LLM)   |
| `05_tasks.py`        | 3     | Defines 3 sequential tasks                    |
| `06_crew.py`         | 3     | Wires the crew together with memory           |
| `orchestrator.py`    | 4     | Core orchestration logic (routing + memory)   |
| `cli.py`             | 4     | Terminal entrypoint for orchestrator          |
| `app.py`             | 4     | Streamlit UI adapter                           |
| `store_policies.txt` | —     | RAG knowledge base (store policies)           |

## 🧠 Concepts Covered

- **Reasoning**: ReAct loop (Thought → Action → Observation)
- **Tool Use**: Web Search + RAG tools
- **Function Calling**: `@tool` decorator → JSON schema → LLM invocation
- **RAG**: Chunk → Embed → Retrieve → Augment (FAISS + sentence-transformers)
- **Memory**: `Crew(memory=True)` for context persistence
- **Multi-Agent**: 3-agent sequential pipeline
- **Orchestration**: LLM-based query routing

## 📚 Tech Stack

- **Framework**: CrewAI
- **Web Search**: Serper API (Google search results)
- **Primary Model (default in code)**: `llama-3.1-8b-instant` (via Groq)
- **Fallback Model**: `llama-3.3-70b-versatile` (via Groq)
- **Model Chain Override**: set `LLM_MODEL_CANDIDATES` in `.env`
- **Alternative**: `deepseek-r1-distill-llama-70b` (for reasoning tasks)
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers, local)
- **Vector Store**: FAISS (in-memory)
- **UI**: Streamlit
