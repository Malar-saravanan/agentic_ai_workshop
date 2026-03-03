"""
Phase 4: Policy Advisor Agent — single agent with RAG tool.

Run:  python 04_policy_advisor.py

What this teaches:
  1. Attach a RAG tool (FAISS vector search) to an agent
  2. The agent DECIDES when to look up store policies (ReAct loop)
  3. Compare: 03 searched the web, this one searches private documents
"""

# ── Setup ───────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import os, sys, logging

os.environ["CREWAI_TRACING_ENABLED"] = "false"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from llm_config import build_llm, run_with_retry

load_dotenv()

# Import RAG tool from 02_tools
from importlib import import_module
tools = import_module("02_tools")
store_policy_rag_tool = tools.store_policy_rag_tool

# ── Step 1: Create the LLM ─────────────────────────────────────────

llm = build_llm(
    model=os.getenv("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
    temperature=0.7,
    max_tokens=500,
    force_react=True,
)

# ── Step 2: Define the Policy Advisor Agent with RAG ────────────────

advisor = Agent(
    role="Store Policy Advisor",
    goal="Answer customer questions about store policies — returns, "
         "warranty, shipping, loyalty rewards, and price matching.",
    backstory="You are ShopWise Electronics' policy expert who knows every "
              "return rule, warranty term, and loyalty benefit. You always "
              "look up the official policy before answering.",
    tools=[store_policy_rag_tool],
    llm=llm,
    max_iter=3,
    verbose=True,
)

# ── Step 3: Chat loop with memory ──────────────────────────────────

history = []


def format_history():
    if not history:
        return "(Start of conversation.)"
    return "\n".join(
        f"Customer: {t['user']}\nAgent: {t['agent']}" for t in history
    )


def ask(user_input):
    def go():
        task = Task(
            description=(
                f"Conversation so far:\n{format_history()}\n\n"
                f"Customer says: {user_input}\n\n"
                f"Look up store policies using your tool to give accurate answers. "
                f"Use conversation history for follow-ups."
            ),
            expected_output="A helpful response based on official store policies.",
            agent=advisor,
        )
        crew = Crew(agents=[advisor], tasks=[task], verbose=True)
        return crew.kickoff()
    return str(run_with_retry(go, llm=llm))


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Policy Advisor Agent — with RAG tool")
    print(f"  Model: {llm.model}  |  Agent: {advisor.role}")
    print("=" * 60)
    print("Ask about store policies. 'clear' = reset, 'quit' = exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "clear":
            history.clear()
            print("[Memory cleared]\n")
            continue

        try:
            response = ask(user_input)
        except Exception as e:
            print(f"\n  [Error: {e}]\n  Wait a moment and try again.\n")
            continue
        print(f"\nAgent: {response}\n")

        history.append({"user": user_input, "agent": response})
        history[:] = history[-10:]

    print("Goodbye!")
    os._exit(0)
