import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, TypedDict


from dotenv import load_dotenv
import google.generativeai as genai
from langgraph.graph import StateGraph, END

from retrieval.generator import stream_answer
from retrieval.company_tool import handle_company_query
from retrieval.context_assembly import assemble_context
from retrieval.history_tool import handle_history_query
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)

ROUTER_MODEL_NAME = "gemini-2.5-flash-lite"
_router_model = None
_graph = None


class AgentState(TypedDict):
    query: str
    conversation_id: Optional[str]
    decision: dict
    history_output: Optional[dict]
    company_output: Optional[dict]
    answer: str
    direct_answer: str


# def run_agent(query, conversation_id=None):
#     query_text = _normalize_query(query)
#     if not query_text:
#         raise ValueError("query must not be empty")

#     graph = _get_graph()
#     final_state = graph.invoke({
#         "query": query_text,
#         "conversation_id": conversation_id,
#         "decision": {},
#         "history_output": None,
#         "company_output": None,
#         "answer": "",
#     })

#     return final_state["answer"]

def run_agent_stream(query, conversation_id=None):
    query_text = _normalize_query(query)

    if not query_text:
        raise ValueError("query must not be empty")

    graph = _get_graph()

    final_state = graph.invoke(
        {
            "query": query_text,
            "conversation_id": conversation_id,
            "decision": {},
            "history_output": None,
            "company_output": None,
            "answer": "",
            "direct_answer": "",
        }
    )

    decision = final_state["decision"]

    # direct response path
    if not decision["needs_tools"]:
        yield from stream_answer(
            direct_answer=(final_state["direct_answer"] or _fallback_direct_answer()))
        return

    assembled_context = assemble_context(
        original_query=decision["original_user_query"],
        normalized_query=decision["normalized_company_query"],
        history_tool_output=final_state["history_output"],
        company_tool_output=final_state["company_output"],
        conversation_id=final_state["conversation_id"],
    )

    assembled_context["prompt"] += (
        f"\n\n---\n\nRespond in the user's language: "
        f"{decision['user_language']}."
    )

    yield from stream_answer(assembled_context=assembled_context)

def _get_graph():
    global _graph

    if _graph is not None:
        return _graph

    _graph = _build_graph()

    logger.info("Compiled LangGraph")

    return _graph

def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", _route)
    graph.add_node("run_tools", _run_tools)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        lambda state: "run_tools" if state["decision"]["needs_tools"] else "end",
        {"run_tools": "run_tools", "end": END},
    )
    graph.add_edge("run_tools", END)

    return graph.compile()


def _route(state):
    decision = _call_router(state["query"])
    decision = _normalize_decision(decision, state["query"])
    logger.info("Router decision: %s", decision)

    state["decision"] = decision
    if not decision["needs_tools"]:
        state["direct_answer"] = (decision["direct_answer"].strip() or _fallback_direct_answer())

    return state


def _run_tools(state):
    decision = state["decision"]

    history_future = None
    company_future = None

    with ThreadPoolExecutor(max_workers=2) as executor:

        if decision["history_needed"]:
            history_future = executor.submit(
                handle_history_query,
                conversation_id=state["conversation_id"],
                exact_range=decision["exact_range"], 
                start=decision["start"] or None, 
                end=decision["end"] or None, 
                )

        if decision["company_needed"]:
            company_future = executor.submit(
                handle_company_query,
                decision["normalized_company_query"],
                state["conversation_id"],
            )

        if history_future is not None:
            state["history_output"] = history_future.result()

        if company_future is not None:
            state["company_output"] = company_future.result()

    return state



def _call_router(query):
    prompt = _build_router_prompt(query)
    model = _get_router_model()

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Router LLM call failed: {exc}") from exc

    return _parse_router_response(response, query)


def _build_router_prompt(query):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
    "You are the routing brain of a multilingual company assistant.\n"
    "Read the user's query and decide what to do next.\n\n"
    f"Current UTC time: {now}\n\n"
    "Decide the following:\n\n"

    "1. needs_tools: true if answering requires company documents or previous conversation history. "
    "false for greetings, thanks, small talk, capability questions, or anything you can answer directly without company data or history. "
    "Never provide passwords, passcodes, credentials, secrets, or sensitive information. Politely refuse such requests and place the refusal inside direct_answer.\n"

    "2. direct_answer: only if needs_tools is false. Write the complete final answer here in the same language and tone used by the user. "
    "Keep greetings and thank-you responses short. "
    "If the user asks what you can do, explain that you are a multilingual company assistant that can answer questions from company documents and previous conversations. "
    "Do not claim capabilities that are unavailable. "
    "Leave this empty if needs_tools is true.\n"

    "3. company_needed: true if the query needs information from company documents such as policies, procedures, products, or other company-specific knowledge.\n"

    "4. history_needed: true if the query refers to previous conversations or earlier discussions.\n"

    "5. normalized_company_query: only if company_needed is true. Rewrite the user's query as a clean English sentence suitable for retrieval. "
    "Preserve meaning exactly. Do not add information, assumptions, bullet points, or unnecessary punctuation. "
    "Leave empty if company_needed is false.\n"

    "6. exact_range: only relevant if history_needed is true. "
    "Set true if the user refers to a specific time period such as today, yesterday, last week, explicit dates, or relative periods like '3 days ago'. "
    "Set false for vague references such as earlier, before, previously, summarize what we discussed, or what did we talk about.\n"

    "7. start and end: only if history_needed is true and exact_range is true. "
    "Compute the UTC date-time range implied by the query relative to the current UTC time above. "
    "Use the exact format 'YYYY-MM-DD HH:MM UTC'. "
    "Leave both empty otherwise.\n"

    "8. user_language: detect the language or mixed style used by the user, for example English, Hindi, Hinglish, Spanish, or mixed.\n"

    "9. original_user_query: copy the user's query exactly as written without translation, rewriting, or modification. "
    "This value will later be passed to the final answer generator so the answer can be produced naturally in the user's language.\n\n"

    "Respond with ONLY a JSON object and exactly these keys:\n"
    "{\n"
    '  "needs_tools": true or false,\n'
    '  "direct_answer": "",\n'
    '  "company_needed": true or false,\n'
    '  "history_needed": true or false,\n'
    '  "normalized_company_query": "",\n'
    '  "exact_range": true or false,\n'
    '  "start": "",\n'
    '  "end": "",\n'
    '  "user_language": "",\n'
    '  "original_user_query": ""\n'
    "}\n\n"
    f"User query: {query}"
        )

def _parse_router_response(response, query):
    text = _extract_text(response)

    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        logger.warning("Router returned invalid JSON, falling back to company tool: %s", exc)
        return _fallback_decision(query)

    return {
        "needs_tools": bool(data.get("needs_tools", True)),
        "direct_answer": str(data.get("direct_answer") or ""),
        "company_needed": bool(data.get("company_needed", True)),
        "history_needed": bool(data.get("history_needed", False)),
        "normalized_company_query": str(data.get("normalized_company_query") or ""),
        "exact_range": bool(data.get("exact_range", False)),
        "start": str(data.get("start") or ""),
        "end": str(data.get("end") or ""),
        "user_language": str(data.get("user_language") or "English"),
        "original_user_query": str(data.get("original_user_query") or query),
    }


def _fallback_decision(query):
    return {
        "needs_tools": True,
        "direct_answer": "",
        "company_needed": True,
        "history_needed": False,
        "normalized_company_query": "",
        "exact_range": False,
        "start": "",
        "end": "",
        "user_language": "English",
        "original_user_query": query,
    }


def _normalize_decision(decision, query):
    if decision["needs_tools"] and not decision["company_needed"] and not decision["history_needed"]:
        decision["company_needed"] = True

    if decision["company_needed"] and not decision["normalized_company_query"]:
        decision["normalized_company_query"] = query

    if not decision["original_user_query"]:
        decision["original_user_query"] = query

    return decision


def _fallback_direct_answer():
    return "How can I help you today?"


def _get_router_model():
    global _router_model

    if _router_model is not None:
        return _router_model

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing required environment variable: GEMINI_API_KEY")

    genai.configure(api_key=api_key)

    try:
        _router_model = genai.GenerativeModel(model_name=ROUTER_MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize router model: {exc}") from exc

    return _router_model


def _extract_text(response):
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                return str(part_text).strip()

    return ""


def _normalize_query(query):
    if query is None:
        return ""
    if not isinstance(query, str):
        query = str(query)
    return query.strip()