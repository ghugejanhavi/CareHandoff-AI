"""
Clinical Q&A — Four RAG Strategies
Implements the professor's notebook pattern:
  1. Naive RAG       — direct retrieval + single LLM call
  2. Intelligent RAG — 3-tool sequential pipeline (search → extract → generate)
  3. Supervisor RAG  — iterative self-evaluation loop (max 3 rounds)
  4. Multi-Agent RAG — 3 parallel specialists + synthesizer

Exposed via:
  MetaIntelligentClinicalRAG — LLM-based router with keyword fallback
  GuardedClinicalRAG         — input + output guardrails wrapping MetaIntelligentClinicalRAG
"""

import logging
from typing import List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_MODEL, LLM_TEMPERATURE
from rag.vector_store import HealthcareVectorStore

logger = logging.getLogger(__name__)

# ── Routing keyword map ────────────────────────────────────────────────────────

STRATEGY_KEYWORDS = {
    "naive": ["what is", "define", "list", "name", "what are", "who is"],
    "intelligent": ["guidelines for", "protocol", "requirements", "how should", "steps for", "criteria", "procedure"],
    "supervisor": ["is it required", "should", "sufficient", "compliant", "appropriate", "necessary", "must"],
    "multi_agent": ["all requirements", "comprehensive", "compare", "every", "across", "full overview", "summarize all"],
}


# ── 1. Naive RAG ───────────────────────────────────────────────────────────────

def naive_rag(query: str, vs: HealthcareVectorStore) -> str:
    """Direct retrieval + single LLM call. Fast, minimal reasoning."""
    docs = vs.retrieve_guidelines(query, k=5)
    if not docs:
        return "No relevant clinical guidelines found for this query."

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    messages = [
        SystemMessage(content=(
            "You are a clinical documentation assistant. Answer questions about clinical "
            "guidelines and documentation standards concisely and accurately. "
            "Do not give personal medical advice."
        )),
        HumanMessage(content=(
            f"Answer this clinical documentation question based on the guidelines below.\n\n"
            f"Guidelines:\n{context}\n\n"
            f"Question: {query}"
        )),
    ]
    return llm.invoke(messages).content


# ── 2. Intelligent RAG ─────────────────────────────────────────────────────────

def intelligent_rag(query: str, vs: HealthcareVectorStore) -> str:
    """3-tool sequential pipeline: search → extract key facts → generate structured answer."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    # Tool 1: Retrieve guidelines
    docs = vs.retrieve_guidelines(query, k=8)
    if not docs:
        return "No relevant clinical guidelines found for this query."
    raw_context = "\n\n---\n\n".join(d.page_content for d in docs)

    # Tool 2: Extract relevant clinical facts
    extract_msgs = [
        SystemMessage(content=(
            "You are a clinical evidence extractor. From the source text, extract only the "
            "facts, criteria, thresholds, and requirements directly relevant to the question. "
            "Format as concise bullet points."
        )),
        HumanMessage(content=(
            f"Question: {query}\n\nSource text:\n{raw_context}\n\n"
            "Extract the key relevant facts in bullet points:"
        )),
    ]
    extracted_facts = llm.invoke(extract_msgs).content

    # Tool 3: Generate structured answer from extracted facts
    generate_msgs = [
        SystemMessage(content=(
            "You are a clinical documentation assistant. Generate a clear, accurate answer "
            "using the extracted clinical facts. Structure your response as:\n"
            "**Direct Answer:** (1-2 sentences)\n"
            "**Key Requirements/Criteria:** (bullet points)\n"
            "**Documentation Notes:** (any important caveats)\n\n"
            "Do not give personal medical advice."
        )),
        HumanMessage(content=(
            f"Question: {query}\n\nExtracted clinical facts:\n{extracted_facts}\n\n"
            "Provide a structured answer:"
        )),
    ]
    return llm.invoke(generate_msgs).content


# ── 3. Supervisor RAG ──────────────────────────────────────────────────────────

def supervisor_rag(query: str, vs: HealthcareVectorStore, max_iterations: int = 3) -> str:
    """Iterative self-evaluation loop: generate → supervisor evaluates → refine (max 3 rounds)."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    docs = vs.retrieve_guidelines(query, k=8)
    if not docs:
        return "No relevant clinical guidelines found for this query."
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # Initial generation
    gen_msgs = [
        SystemMessage(content=(
            "You are a clinical documentation expert. Answer questions about clinical guidelines "
            "with precision and completeness."
        )),
        HumanMessage(content=(
            f"Guidelines:\n{context}\n\nQuestion: {query}\n\nProvide a comprehensive answer:"
        )),
    ]
    answer = llm.invoke(gen_msgs).content

    for iteration in range(max_iterations - 1):
        # Supervisor evaluation
        eval_msgs = [
            SystemMessage(content=(
                "You are a quality supervisor evaluating a clinical documentation answer. "
                "Respond with ONLY 'SUFFICIENT' if the answer is complete and accurate, "
                "or 'INSUFFICIENT: <specific gaps>' if critical information is missing."
            )),
            HumanMessage(content=(
                f"Question: {query}\n\nAnswer to evaluate:\n{answer}\n\n"
                f"Available guidelines:\n{context}\n\n"
                "Is this answer sufficient? Respond SUFFICIENT or INSUFFICIENT: <gaps>"
            )),
        ]
        evaluation = llm.invoke(eval_msgs).content.strip()

        if evaluation.upper().startswith("SUFFICIENT"):
            logger.info("[SupervisorRAG] Accepted at iteration %d", iteration + 1)
            break

        gaps = evaluation.replace("INSUFFICIENT:", "").strip()
        refine_msgs = [
            SystemMessage(content="You are a clinical documentation expert. Improve the answer by addressing the identified gaps."),
            HumanMessage(content=(
                f"Question: {query}\n\nCurrent answer:\n{answer}\n\n"
                f"Gaps to address: {gaps}\n\nGuidelines:\n{context}\n\n"
                "Provide an improved, complete answer:"
            )),
        ]
        answer = llm.invoke(refine_msgs).content
        logger.info("[SupervisorRAG] Refined at iteration %d", iteration + 2)

    return answer


# ── 4. Multi-Agent RAG ─────────────────────────────────────────────────────────

def multi_agent_rag(query: str, vs: HealthcareVectorStore) -> str:
    """3 parallel specialist agents (Heart Failure, Hypertension, General) → synthesizer."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    specialists = [
        ("Heart Failure Specialist", "Heart Failure.txt"),
        ("Hypertension Specialist", "Hypertension.txt"),
        ("General Guidelines Specialist", "redtoolkit.txt"),
    ]

    specialist_responses = []

    for agent_name, source_filter in specialists:
        docs = vs.retrieve_guidelines(query, k=4, source_filter=source_filter)
        if not docs:
            docs = vs.retrieve_guidelines(query, k=3)

        if docs:
            context = "\n\n".join(d.page_content for d in docs)
            msgs = [
                SystemMessage(content=(
                    f"You are a {agent_name} focusing on clinical documentation guidelines. "
                    "Answer the question using only the provided guidelines from your specialty. "
                    "Be specific and cite relevant criteria or thresholds."
                )),
                HumanMessage(content=f"Guidelines ({agent_name}):\n{context}\n\nQuestion: {query}"),
            ]
            response = llm.invoke(msgs).content
            specialist_responses.append(f"**{agent_name}**:\n{response}")

    if not specialist_responses:
        return "No relevant guidelines found across specialist areas."

    combined = "\n\n---\n\n".join(specialist_responses)
    synth_msgs = [
        SystemMessage(content=(
            "You are a clinical documentation synthesizer. Combine the specialist responses "
            "into one coherent, comprehensive answer. Identify consensus, resolve conflicts, "
            "and highlight where guidelines agree or differ. Structure as:\n"
            "**Consensus Answer:** \n"
            "**Guideline-Specific Requirements:** \n"
            "**Documentation Checklist:** "
        )),
        HumanMessage(content=(
            f"Question: {query}\n\nSpecialist responses:\n{combined}\n\n"
            "Synthesize a comprehensive answer:"
        )),
    ]
    return llm.invoke(synth_msgs).content


# ── Meta-Intelligent Router ────────────────────────────────────────────────────

class MetaIntelligentClinicalRAG:
    """Routes each query to the optimal RAG strategy using LLM classification with keyword fallback."""

    def __init__(self, vector_store: HealthcareVectorStore):
        self.vs = vector_store
        self._llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)

    def analyze_query(self, query: str) -> str:
        """LLM-based strategy classification with keyword fallback."""
        try:
            msgs = [
                SystemMessage(content=(
                    "You are a query router for a clinical documentation Q&A system. "
                    "Classify the query into exactly one strategy:\n"
                    "- naive: Simple factual lookups (definitions, lists, 'what is')\n"
                    "- intelligent: Protocol or guideline questions requiring step-by-step reasoning\n"
                    "- supervisor: Compliance/sufficiency checks ('should', 'required', 'is it compliant')\n"
                    "- multi_agent: Comprehensive questions spanning multiple clinical guideline areas\n\n"
                    "Respond with ONLY the strategy name: naive, intelligent, supervisor, or multi_agent"
                )),
                HumanMessage(content=f"Query: {query}"),
            ]
            result = self._llm.invoke(msgs).content.strip().lower()
            if result in ("naive", "intelligent", "supervisor", "multi_agent"):
                logger.info("[MetaRouter] LLM classified as: %s", result)
                return result
        except Exception as e:
            logger.warning("[MetaRouter] LLM classification failed: %s — using keyword fallback", e)

        return self._fallback(query)

    def _fallback(self, query: str) -> str:
        """Keyword-based fallback classification."""
        q_lower = query.lower()
        for strategy, keywords in STRATEGY_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                logger.info("[MetaRouter] Keyword fallback → %s", strategy)
                return strategy
        return "intelligent"

    def execute(self, query: str, strategy: str) -> str:
        """Dispatch to the correct strategy function."""
        if strategy == "naive":
            return naive_rag(query, self.vs)
        elif strategy == "intelligent":
            return intelligent_rag(query, self.vs)
        elif strategy == "supervisor":
            return supervisor_rag(query, self.vs)
        elif strategy == "multi_agent":
            return multi_agent_rag(query, self.vs)
        return intelligent_rag(query, self.vs)

    def enhance(self, answer: str, strategy: str) -> str:
        """Append strategy indicator to the answer."""
        labels = {
            "naive": "Naive RAG",
            "intelligent": "Intelligent RAG",
            "supervisor": "Supervisor RAG",
            "multi_agent": "Multi-Agent RAG",
        }
        label = labels.get(strategy, strategy)
        return f"{answer}\n\n---\n*Strategy: **{label}** | Source: Clinical Guidelines Knowledge Base*"

    def run(self, query: str) -> Tuple[str, str]:
        """Full pipeline: classify → execute → enhance. Returns (enhanced_answer, strategy)."""
        strategy = self.analyze_query(query)
        answer = self.execute(query, strategy)
        enhanced = self.enhance(answer, strategy)
        return enhanced, strategy


# ── Guardrails ─────────────────────────────────────────────────────────────────

# Dedicated low-temperature LLM for guardrail checks (fast, deterministic)
_llm_guard = ChatOpenAI(model=LLM_MODEL, temperature=0)


class GuardedClinicalRAG:
    """
    Production-safe wrapper following the professor's guardrail pattern.

    Input pipeline  (cheapest → most expensive):
      1. check_topic_allowed  — keyword block list (no LLM call)
      2. check_prompt_injection — injection pattern list (no LLM call)
      3. check_on_topic       — LLM relevance check (fails open)

    Output pipeline:
      1. check_sensitive_info — keyword scan for PII/sensitive fields

    Orchestrated by guarded_query(); exposed to Gradio via chat().
    """

    def __init__(self, meta_rag: MetaIntelligentClinicalRAG):
        self.meta_rag = meta_rag
        self.blocked_topics = [
            "hack", "exploit", "illegal", "weapon",
            "cure me", "prescribe me", "diagnose me", "treat me",
            "cure my", "treat my", "my medication", "my prescription",
        ]
        self.sensitive_fields = [
            "ssn", "social security", "home address", "personal phone",
            "date of birth", "medical record number", "mrn",
        ]

    # ── Input guardrail 1: blocked topic keywords ──────────────────────────

    def check_topic_allowed(self, query: str) -> dict:
        q = query.lower()
        for topic in self.blocked_topics:
            if topic in q:
                return {"allowed": False, "reason": f"Blocked topic detected: '{topic}'"}
        return {"allowed": True, "reason": None}

    # ── Input guardrail 2: prompt injection patterns ───────────────────────

    def check_prompt_injection(self, query: str) -> dict:
        injection_patterns = [
            "ignore previous instructions",
            "ignore above instructions",
            "disregard your instructions",
            "forget your rules",
            "you are now",
            "pretend to be",
            "system prompt",
            "reveal your instructions",
            "act as",
            "jailbreak",
        ]
        q = query.lower()
        for pattern in injection_patterns:
            if pattern in q:
                return {"allowed": False, "reason": "Prompt injection detected"}
        return {"allowed": True, "reason": None}

    # ── Input guardrail 3: LLM on-topic relevance check ───────────────────

    def check_on_topic(self, query: str) -> dict:
        """LLM check — fails open so a network error never blocks a legitimate user."""
        prompt = (
            "You are evaluating whether a query is appropriate for a clinical DOCUMENTATION assistant.\n\n"
            "This system ONLY answers questions about:\n"
            "- Clinical documentation standards and guidelines\n"
            "- Discharge note requirements and best practices\n"
            "- Protocol and guideline compliance for documentation\n"
            "- What should be documented in clinical notes\n\n"
            "This system does NOT answer:\n"
            "- Personal medical advice (e.g. 'how do I cure X', 'what drug should I take')\n"
            "- Personal diagnosis requests (e.g. 'do I have hypertension')\n"
            "- Personal treatment questions (e.g. 'how can I cure hypertension')\n"
            "- Completely off-topic subjects (recipes, sports, entertainment)\n\n"
            f'Query: "{query}"\n\n'
            "Is this query appropriate for a clinical documentation assistant?\n"
            "Answer ONLY \"yes\" or \"no\"."
        )
        try:
            response = _llm_guard.invoke([{"role": "user", "content": prompt}])
            if response.content.strip().lower() == "yes":
                return {"allowed": True, "reason": None}
            return {"allowed": False, "reason": "Off-topic or personal medical advice query"}
        except Exception:
            return {"allowed": True, "reason": None}  # fail open

    # ── Run all input guardrails (cheapest first) ──────────────────────────

    def run_input_guardrails(self, query: str) -> dict:
        topic_check = self.check_topic_allowed(query)
        if not topic_check["allowed"]:
            return topic_check

        injection_check = self.check_prompt_injection(query)
        if not injection_check["allowed"]:
            return injection_check

        relevance_check = self.check_on_topic(query)
        if not relevance_check["allowed"]:
            return relevance_check

        return {"allowed": True, "reason": None}

    # ── Output guardrail: sensitive field scan ─────────────────────────────

    def check_sensitive_info(self, response: str) -> dict:
        r = response.lower()
        for field in self.sensitive_fields:
            if field in r:
                return {"passed": False, "reason": f"Contains sensitive info: '{field}'"}
        return {"passed": True, "reason": None}

    def run_output_guardrails(self, query: str, response: str) -> dict:
        sensitive_check = self.check_sensitive_info(response)
        if not sensitive_check["passed"]:
            return sensitive_check
        return {"passed": True, "reason": None}

    # ── Main orchestrator ──────────────────────────────────────────────────

    def guarded_query(self, query: str) -> dict:
        """Run input guardrails → RAG → output guardrails. Returns structured result dict."""
        input_check = self.run_input_guardrails(query)
        if not input_check["allowed"]:
            return {
                "response": f"I can't process this query. Reason: {input_check['reason']}",
                "blocked": True,
                "stage": "input",
            }

        try:
            answer, strategy = self.meta_rag.run(query)
        except Exception as e:
            logger.error("[GuardedRAG] Execution error: %s", e)
            return {
                "response": "An error occurred while processing your query.",
                "blocked": True,
                "stage": "execution",
            }

        output_check = self.run_output_guardrails(query, answer)
        if not output_check["passed"]:
            return {
                "response": f"I found relevant information but cannot share it. Reason: {output_check['reason']}",
                "blocked": True,
                "stage": "output",
            }

        return {
            "response": answer,
            "strategy": strategy,
            "blocked": False,
        }

    # ── Gradio-compatible chat function ────────────────────────────────────

    def chat(self, message: str, history: list) -> str:
        """Entry point for gr.ChatInterface."""
        result = self.guarded_query(message)
        if result["blocked"]:
            return (
                f"**BLOCKED:** {result['response']}\n\n"
                f"*[Blocked at stage: {result['stage']}]*"
            )
        label = result["strategy"].replace("_", " ").title()
        return f"{result['response']}\n\n*[{label} Strategy | Guardrails passed]*"
