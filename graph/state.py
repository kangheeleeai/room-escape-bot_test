from typing import Optional, TypedDict


class BotState(TypedDict, total=False):
    # Per-turn input
    user_query: str
    user_context: Optional[str]

    # Intent analysis output
    intent: dict
    action: str

    # Filter resolution
    filters: dict
    exclude_ids: list
    target_users: list

    # Recommendation results (parallel branches write to disjoint keys)
    results_rule: list
    results_personalized: list
    results_text: list
    results_web: list

    # Final output
    reply_text: str
    debug_info: dict

    # Persisted across turns by MemorySaver checkpointer
    shown_ids: set
    last_filters: dict
