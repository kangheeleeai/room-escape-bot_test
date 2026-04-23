import logging
from typing import Optional

import streamlit as st
from tavily import TavilyClient

log = logging.getLogger(__name__)


@st.cache_resource
def make_tavily_client(api_key: str) -> Optional[TavilyClient]:
    if not api_key:
        return None
    try:
        return TavilyClient(api_key=api_key)
    except Exception as e:
        log.warning(f"Tavily 클라이언트 초기화 실패: {e}")
        return None


def search_web(client: Optional[TavilyClient], query: str, max_results: int = 5) -> list[dict]:
    """Tavily 검색 실행. 결과 형식: [{'title','content','url'}, ...]"""
    if not client or not query:
        return []
    try:
        response = client.search(
            query=f"방탈출 {query}",
            max_results=max_results,
            search_depth="basic",
        )
        return response.get("results", []) or []
    except Exception as e:
        log.warning(f"Tavily 검색 실패: {e}")
        return []


def format_snippets(results: list[dict], max_chars: int = 200) -> str:
    """LLM 입력용 snippet 텍스트로 직렬화."""
    lines = []
    for r in results:
        title = (r.get("title") or "").strip()
        content = (r.get("content") or "").strip()[:max_chars]
        if title or content:
            lines.append(f"- {title}: {content}")
    return "\n".join(lines)
