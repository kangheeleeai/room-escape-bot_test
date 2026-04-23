from typing import Literal, Optional

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

MODEL_NAME = "llama-3.1-8b-instant"


class HistoryItem(BaseModel):
    theme: str = Field(description="Theme name in Korean")
    location: Optional[str] = Field(default=None, description="Location name (e.g., 강남)")


class IntentResult(BaseModel):
    """Structured intent extracted from a Korean Escape Room query."""

    action: Literal[
        "played_check_inquiry",
        "played_check",
        "not_played_check",
        "recommend",
        "another_recommend",
    ] = Field(description="User's intent classification")
    keywords: list[str] = Field(
        default_factory=list,
        description="Genre/vibe keywords. EXCLUDE location names, rating numbers, and person counts.",
    )
    min_rating: Optional[float] = Field(
        default=None,
        description="Minimum rating threshold (e.g., 4.0 for '4점 이상', 4.5 for '4.5 넘는거').",
    )
    people_count: Optional[int] = Field(
        default=None,
        description="Group size (e.g., 2 for '둘이서', 4 for '4명' or '4인').",
    )
    mentioned_users: list[str] = Field(
        default_factory=list,
        description="Other users explicitly mentioned in the query.",
    )
    items: list[HistoryItem] = Field(
        default_factory=list,
        description="Played-theme entries; populate ONLY for played_check / not_played_check actions.",
    )


class ThemeNameList(BaseModel):
    """웹 검색 snippet에서 추출한 한국 방탈출 테마명 리스트."""

    theme_names: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 10 unique Korean Escape Room THEME names mentioned in the snippets. "
            "Exclude store/brand names, locations, descriptions, and English-only words."
        ),
    )


THEME_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract Korean Escape Room theme names from web search snippets. "
            "Return ONLY theme names — no store names, no descriptions, no locations.",
        ),
        (
            "user",
            'User query: "{user_query}"\n\n'
            "Web snippets:\n{snippets}\n\n"
            "Extract up to 10 distinct theme names that match the user's interest.",
        ),
    ]
)


INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an Escape Room recommendation intent analyzer. "
            "Always interpret Korean queries. Return only the structured fields requested.",
        ),
        (
            "user",
            'Analyze the following Korean Escape Room query and classify the user intent.\n\n'
            'Query: "{user_query}"\n\n'
            "Action rules:\n"
            "1. played_check_inquiry: Asking how to record played themes.\n"
            "2. played_check: User states they played a theme (e.g. '강남 링 했어').\n"
            "3. not_played_check: User wants to cancel/remove a record.\n"
            "4. recommend: User asks for recommendations.\n"
            "5. another_recommend: Asking for other options ('다른거 추천해줘').",
        ),
    ]
)


@st.cache_resource
def make_llm(api_key: str) -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, temperature=0.1, api_key=api_key)


@st.cache_resource
def make_intent_chain(api_key: str):
    llm = make_llm(api_key)
    return INTENT_PROMPT | llm.with_structured_output(IntentResult)


@st.cache_resource
def make_theme_extract_chain(api_key: str):
    llm = make_llm(api_key)
    return THEME_EXTRACT_PROMPT | llm.with_structured_output(ThemeNameList)
