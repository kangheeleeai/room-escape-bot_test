import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import make_nodes
from .state import BotState


@st.cache_resource
def compile_bot_graph(
    _db,
    _vector_recommender,
    _rule_recommender,
    api_key: str,
    tavily_api_key: str = "",
):
    """LangGraph 컴파일 + MemorySaver 부착. Streamlit 인스턴스에서 1회만 빌드된다.

    `_db`, `_vector_recommender`, `_rule_recommender`는 해시 불가 객체이므로
    Streamlit 캐시 키에서 제외하기 위해 언더스코어 prefix를 사용한다.
    """
    nodes = make_nodes(
        {
            "api_key": api_key,
            "tavily_api_key": tavily_api_key,
            "db": _db,
            "vector_recommender": _vector_recommender,
            "rule_recommender": _rule_recommender,
        }
    )

    g = StateGraph(BotState)

    # 노드 등록
    g.add_node("init_turn", nodes["init_turn"])
    g.add_node("analyze_intent", nodes["analyze_intent"])
    g.add_node("handle_inquiry", nodes["handle_inquiry"])
    g.add_node("handle_history", nodes["handle_history"])
    g.add_node("build_filters", nodes["build_filters"])
    g.add_node("recommend_rule", nodes["recommend_rule"])
    g.add_node("recommend_personalized", nodes["recommend_personalized"])
    g.add_node("recommend_web_search", nodes["recommend_web_search"])
    g.add_node("aggregate", nodes["aggregate"])
    g.add_node("recommend_text_fallback", nodes["recommend_text_fallback"])
    g.add_node("compose_reply", nodes["compose_reply"])

    # START → init_turn → analyze_intent
    g.add_edge(START, "init_turn")
    g.add_edge("init_turn", "analyze_intent")

    # 의도 기반 분기
    g.add_conditional_edges(
        "analyze_intent",
        nodes["route_after_intent"],
        {
            "inquiry": "handle_inquiry",
            "history": "handle_history",
            "recommend": "build_filters",
        },
    )
    g.add_edge("handle_inquiry", END)
    g.add_edge("handle_history", END)

    # 추천: rule + personalized + web 3-way 병렬 실행 → aggregate fan-in
    g.add_edge("build_filters", "recommend_rule")
    g.add_edge("build_filters", "recommend_personalized")
    g.add_edge("build_filters", "recommend_web_search")
    g.add_edge("recommend_rule", "aggregate")
    g.add_edge("recommend_personalized", "aggregate")
    g.add_edge("recommend_web_search", "aggregate")

    # 둘 다 비면 텍스트 벡터 폴백
    g.add_conditional_edges(
        "aggregate",
        nodes["route_after_aggregate"],
        {
            "fallback": "recommend_text_fallback",
            "compose": "compose_reply",
        },
    )
    g.add_edge("recommend_text_fallback", "compose_reply")
    g.add_edge("compose_reply", END)

    return g.compile(checkpointer=MemorySaver())
