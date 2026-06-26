import json
import logging
import os
import time
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
# "gemma-4-26b-a4b-it"
ROUTER_MODEL_NAME = "gemini-flash-lite-latest"
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



def run_agent_stream(query, conversation_id=None):
    query_text = _normalize_query(query)

    if not query_text:
        raise ValueError("query must not be empty")

    graph = _get_graph()

    graph_start = time.perf_counter()

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
    graph_time = time.perf_counter() - graph_start
    print(f"[TIMER] Graph invoke total: {graph_time:.3f}s")

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
    start = time.perf_counter()

    decision = _call_router(state["query"])

    router_time = time.perf_counter() - start
    print(f"[TIMER] Router LLM: {router_time:.3f}s")

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
            history_start = time.perf_counter()
            state["history_output"] = history_future.result()
            history_time = time.perf_counter() - history_start
            print(f"[TIMER] History Tool: {history_time:.3f}s")

        if company_future is not None:
            company_start = time.perf_counter()
            state["company_output"] = company_future.result()
            company_time = time.perf_counter() - company_start
            print(f"[TIMER] Company Tool Total: {company_time:.3f}s")

    return state



def _call_router(query):
    t0 = time.perf_counter()
    prompt = _build_router_prompt(query)
    print(f"[TIMER] Router Prompt Build: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    model = _get_router_model()
    print(f"[TIMER] Router Model Fetch: {time.perf_counter() - t0:.3f}s")

    try:
        t0 = time.perf_counter()
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
                max_output_tokens=300,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "needs_tools": {"type": "boolean"},
                        "direct_answer": {"type": "string"},
                        "company_needed": {"type": "boolean"},
                        "history_needed": {"type": "boolean"},
                        "normalized_company_query": {"type": "string"},
                        "exact_range": {"type": "boolean"},
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "user_language": {"type": "string"},
                        "original_user_query": {"type": "string"},
                    },
                    "required": [
                        "needs_tools",
                        "direct_answer",
                        "company_needed",
                        "history_needed",
                        "normalized_company_query",
                        "exact_range",
                        "start",
                        "end",
                        "user_language",
                        "original_user_query",
                    ],
                },
            ),
        )
        print(f"[TIMER] Router Gemini Generate Content: " f"{time.perf_counter() - t0:.3f}s")
        # usage = getattr(response, "usage_metadata", None)
        # print("USAGE:", usage)
        # print("CANDIDATES: ",response.candidates)
    except Exception as exc:
        raise RuntimeError(f"Router LLM call failed: {exc}") from exc

    return _parse_router_response(response, query)


def _build_router_prompt(query):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"UTC: {now}\n"
        "Route this query for a multilingual company chatbot. Return only VALID JSON, no explanation and no deep analysis.\n"
        "Return the complete JSON object. Every key must be present. Never omit fields. Use empty strings instead of missing values.\n\n"
        "needs_tools: false for greetings, thanks, small talk, capability questions. true otherwise.\n"
        "direct_answer: full reply in user's language if needs_tools=false. Refuse passwords/credentials politely. Empty if needs_tools=true.\n"
        "company_needed: true if company documents (policies, products, procedures) are needed.\n"
        "history_needed: true if previous conversations are needed.\n"
        "normalized_company_query: clean English retrieval sentence if company_needed=true. Empty otherwise.\n"
        "exact_range: true for specific time refs (today, yesterday, last week, X days/hours ago, explicit dates). false for vague refs (earlier, previously, before, what did we discuss).\n"
        "start/end: UTC range if exact_range=true, format 'YYYY-MM-DD HH:MM UTC'. Empty otherwise.\n"
        "user_language: detected language/style (English, Hindi, Hinglish, etc).\n"
        "original_user_query: exact copy of the user query.\n\n"
        "JSON keys: needs_tools, direct_answer, company_needed, history_needed, normalized_company_query, exact_range, start, end, user_language, original_user_query\n\n"
        f"Query: {query}"
    )

def _parse_router_response(response, query):
    t0 = time.perf_counter()
    text = _extract_text(response)
    # print("\n========== ROUTER RAW RESPONSE ==========")
    # print(len(text))
    # print(text)
    # print("=========================================\n")
    print(f"[TIMER] Router Extract Text: "f"{time.perf_counter() - t0:.3f}s")

    try:
        t0 = time.perf_counter()
        data = json.loads(text)
        print(f"[TIMER] Router json.loads: "f"{time.perf_counter() - t0:.6f}s")

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
        "start": _parse_utc_datetime(data.get("start")),
        "end": _parse_utc_datetime(data.get("end")),
        "user_language": str(data.get("user_language") or "English"),
        "original_user_query": str(data.get("original_user_query") or query),
    }

def _parse_utc_datetime(value):
    if not value:
        return None
    try:
        return datetime.strptime(value,"%Y-%m-%d %H:%M UTC",).replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning("Failed to parse router datetime: %s", value,)
        return None

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
    if decision["company_needed"] or decision["history_needed"]:
        decision["needs_tools"] = True

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
        t0 = time.perf_counter()
        _router_model = genai.GenerativeModel(model_name=ROUTER_MODEL_NAME)
        print(f"[TIMER] Router Model Init: "f"{time.perf_counter() - t0:.3f}s")
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