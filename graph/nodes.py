import logging
from typing import Optional

from utils import sort_candidates_by_query

from .keywords import has_recognized_keyword
from .llm import make_intent_chain, make_theme_extract_chain
from .locations import extract_locations
from .state import BotState
from .web_search import format_snippets, make_tavily_client, search_web

log = logging.getLogger(__name__)


def _find_theme_id(db, location: Optional[str], theme_name: str) -> Optional[int]:
    from database import load_themes_snapshot

    try:
        snapshot = load_themes_snapshot(db)
        target = theme_name.replace(" ", "")
        for t in snapshot:
            if location and t["location"] != location:
                continue
            title = t["title"].replace(" ", "")
            letters = t["letters"].replace(" ", "")
            if target in title or (letters and target in letters):
                log.info(f"[DB] 테마 매칭: {t['title']} (id={t['doc_id']})")
                return int(t["ref_id"] or t["doc_id"])
    except Exception as e:
        log.warning(f"find_theme_id 에러: {e}")
    return None


def _update_play_history(db, nickname: str, theme_id: int, action: str) -> str:
    from database import FieldFilter, firestore

    try:
        users_ref = db.collection("users")
        docs = list(
            users_ref.where(filter=FieldFilter("nickname", "==", nickname)).limit(1).stream()
        )
        if not docs:
            return "❌ 유저 미등록"

        user_doc = docs[0]
        if action == "played_check":
            user_doc.reference.update({"played": firestore.ArrayUnion([theme_id])})
            return "추가 완료"
        if action == "not_played_check":
            user_doc.reference.update({"played": firestore.ArrayRemove([theme_id])})
            return "삭제 완료"
        return "알 수 없는 요청"
    except Exception as e:
        return f"에러: {e}"


def _match_themes_in_db(db, candidates: list[str], locations: list[str], exclude_ids: set, limit: int) -> list[dict]:
    """Tavily에서 추출한 테마명 후보를 캐시된 themes 스냅샷에서 substring 매칭."""
    from database import load_themes_snapshot

    if not candidates:
        return []

    norm_cands = [c.replace(" ", "") for c in candidates if c]
    matched: list[dict] = []
    seen_ids: set[str] = set()

    try:
        snapshot = load_themes_snapshot(db)
        for t in snapshot:
            if len(matched) >= limit * 5:
                break
            if locations and t["location"] not in locations:
                continue
            title = t["title"].replace(" ", "")
            letters = t["letters"].replace(" ", "")
            if not any(c in title or (letters and c in letters) for c in norm_cands):
                continue

            try:
                tid = int(t["ref_id"] or t["doc_id"])
                if tid in exclude_ids or str(tid) in exclude_ids:
                    continue
            except Exception:
                if t["doc_id"] in exclude_ids:
                    continue

            if t["doc_id"] in seen_ids:
                continue
            seen_ids.add(t["doc_id"])

            matched.append(
                {
                    "id": t["doc_id"],
                    "title": t["title"],
                    "store": t["store_name"],
                    "location": t["location"],
                    "desc": t["description"][:150],
                    "rating": t["satisfyTotalRating"],
                    "fear": t["fearTotalRating"],
                    "activity": t["activityTotalRating"],
                    "difficulty": t["difficultyTotalRating"],
                }
            )
    except Exception as e:
        log.warning(f"[Web] DB 매칭 실패: {e}")

    matched.sort(key=lambda x: x.get("rating", 0), reverse=True)
    return matched[:limit]


def make_nodes(deps: dict) -> dict:
    """노드 팩토리. deps에 의존성을 클로저로 묶어 그래프에 등록할 수 있는 형태로 반환."""
    intent_chain = make_intent_chain(deps["api_key"])
    theme_extract_chain = make_theme_extract_chain(deps["api_key"])
    tavily_client = make_tavily_client(deps.get("tavily_api_key") or "")
    rule_rec = deps["rule_recommender"]
    vec_rec = deps["vector_recommender"]
    db = deps["db"]

    def init_turn(state: BotState) -> dict:
        # 이전 턴의 results_*가 새어 나오지 않도록 매 턴 시작 시 초기화
        return {
            "results_rule": [],
            "results_personalized": [],
            "results_text": [],
            "results_web": [],
            "intent": {},
            "debug_info": {},
        }

    def analyze_intent(state: BotState) -> dict:
        query = state["user_query"]
        log.info(f"[LLM] 의도 분석: '{query}'")
        try:
            result: "IntentResult" = intent_chain.invoke({"user_query": query})
            intent = result.model_dump()
        except Exception as e:
            log.warning(f"의도 분석 실패 ({e}) → recommend로 폴백")
            intent = {
                "action": "recommend",
                "keywords": [query],
                "min_rating": None,
                "people_count": None,
                "mentioned_users": [],
                "items": [],
            }

        # 지역 추출은 결정론적 매칭으로 처리 (LLM에 맡기지 않음)
        locs = extract_locations(query)
        intent["locations"] = locs

        # played_check items에 location이 비어있으면 추출된 지역으로 보강
        if intent.get("items") and locs:
            for item in intent["items"]:
                if not item.get("location"):
                    item["location"] = locs[0]

        log.info(
            f"   -> action={intent.get('action')}, locations={locs}, "
            f"min_rating={intent.get('min_rating')}, people_count={intent.get('people_count')}"
        )

        return {
            "intent": intent,
            "action": intent.get("action", "recommend"),
            "debug_info": {"intent": intent, "query": query},
        }

    def handle_inquiry(state: BotState) -> dict:
        return {
            "reply_text": "플레이한 테마를 `[지역] [테마명] 했어` 라고 말씀해주시면 기록해 드립니다!",
        }

    def handle_history(state: BotState) -> dict:
        user = state.get("user_context")
        intent = state["intent"]
        action = state["action"]

        if not user:
            return {"reply_text": "⚠️ 닉네임을 먼저 설정해주세요."}

        items = list(intent.get("items") or [])
        if not items:
            return {"reply_text": "⚠️ 어떤 테마를 기록할지 알 수 없습니다."}

        lines = []
        for item in items:
            theme = item.get("theme")
            loc = item.get("location")
            if not theme:
                continue
            tid = _find_theme_id(db, loc, theme)
            if tid:
                res = _update_play_history(db, user, tid, action)
                lines.append(f"- **{theme}**: {res}")
            else:
                lines.append(f"- **{theme}**: ⚠️ 테마 못 찾음")

        return {"reply_text": "\n".join(lines) if lines else "⚠️ 처리할 항목이 없습니다."}

    def build_filters(state: BotState) -> dict:
        intent = state["intent"]
        action = state["action"]
        user_ctx = state.get("user_context") or ""

        current = {
            "locations": intent.get("locations") or [],
            "keywords": intent.get("keywords") or [],
            "mentioned_users": intent.get("mentioned_users") or [],
            "min_rating": intent.get("min_rating"),
            "people_count": intent.get("people_count"),
        }

        # 닉네임 + LLM이 추출한 mentioned_users 합치기
        target_users = [u.strip() for u in user_ctx.split(",") if u.strip()]
        for u in current["mentioned_users"]:
            if u not in target_users:
                target_users.append(u)

        # another_recommend: 직전 필터 + 현재 턴의 위치/평점/인원만 덮어쓰기
        if action == "another_recommend":
            base = dict(state.get("last_filters") or {})
            for k in ("locations", "min_rating", "people_count"):
                if current[k]:
                    base[k] = current[k]
            filters = base
            exclude_ids = list(state.get("shown_ids") or [])
        else:
            filters = current
            exclude_ids = []

        log.info(f"필터 적용: {filters} / 제외 {len(exclude_ids)}개")
        return {
            "filters": filters,
            "exclude_ids": exclude_ids,
            "target_users": target_users,
        }

    def recommend_rule(state: BotState) -> dict:
        users = state.get("target_users") or []
        nicknames = users if len(users) > 1 else (users[0] if users else None)
        results = rule_rec.search_themes(
            state["filters"],
            state.get("user_query", ""),
            limit=3,
            nicknames=nicknames,
            exclude_ids=state.get("exclude_ids") or [],
        )
        return {"results_rule": results or []}

    def recommend_personalized(state: BotState) -> dict:
        users = state.get("target_users") or []
        if not users:
            return {"results_personalized": []}

        nicknames = users if len(users) > 1 else users[0]
        results = vec_rec.recommend_by_user_search(
            nicknames,
            user_query=state.get("user_query", ""),
            limit=3,
            filters=state["filters"],
            exclude_ids=state.get("exclude_ids") or [],
        )
        return {"results_personalized": results or []}

    def recommend_web_search(state: BotState) -> dict:
        """위치 외 인식 키워드가 0개일 때만 동작.
        Tavily 검색 → LLM이 테마명 추출 → Firestore 매칭 → 플레이/노출 제외 → 평점순 추천.
        """
        query = state.get("user_query", "") or ""

        if has_recognized_keyword(query):
            return {"results_web": []}

        if tavily_client is None:
            log.info("[Web] Tavily 키 없음 → 스킵")
            return {"results_web": []}

        log.info(f"[Web] Tavily 검색: '{query}'")
        results = search_web(tavily_client, query, max_results=5)
        if not results:
            return {"results_web": []}

        snippets = format_snippets(results)
        try:
            extracted = theme_extract_chain.invoke(
                {"user_query": query, "snippets": snippets}
            )
            candidates = list(extracted.theme_names or [])
        except Exception as e:
            log.warning(f"[Web] 테마명 추출 실패: {e}")
            return {"results_web": []}

        if not candidates:
            log.info("[Web] 추출된 테마명 없음")
            return {"results_web": []}

        log.info(f"[Web] 추출 {len(candidates)}개: {candidates}")

        locations = (state.get("filters") or {}).get("locations") or []
        exclude_ids = set(state.get("exclude_ids") or [])

        # 플레이 기록 제외 — VectorRecommender 내부 헬퍼 재사용
        users = state.get("target_users") or []
        if users:
            user_ctx = users if len(users) > 1 else users[0]
            try:
                played = vec_rec._get_played_ids_internal(user_ctx)
                exclude_ids |= set(played)
            except Exception as e:
                log.warning(f"[Web] 플레이 기록 조회 실패: {e}")

        matches = _match_themes_in_db(db, candidates, locations, exclude_ids, limit=3)
        log.info(f"[Web] DB 매칭 {len(matches)}개")
        return {"results_web": matches}

    def aggregate(state: BotState) -> dict:
        # 병렬 fan-in 동기화 지점. 상태 변경 없음.
        return {}

    def recommend_text_fallback(state: BotState) -> dict:
        query = state.get("user_query", "")
        results = vec_rec.recommend_by_text(
            query,
            filters=state["filters"],
            exclude_ids=state.get("exclude_ids") or [],
        )
        if results:
            results = sort_candidates_by_query(results, query)[:3]
        return {"results_text": results or []}

    def compose_reply(state: BotState) -> dict:
        intent = state["intent"]
        action = state["action"]
        filters = state["filters"]
        user_ctx = state.get("user_context")

        rule = state.get("results_rule") or []
        pers = state.get("results_personalized") or []
        text = state.get("results_text") or []
        web = state.get("results_web") or []

        if not (rule or pers or text or web):
            return {"reply_text": "조건에 맞는 테마를 찾지 못했습니다."}

        # 새 추천이면 shown_ids 리셋, 'another_recommend'면 누적
        new_shown = set() if action == "recommend" else set(state.get("shown_ids") or [])
        for lst in (rule, pers, text, web):
            for c in lst:
                if "id" in c:
                    new_shown.add(c["id"])

        locs = intent.get("locations") or []
        if len(locs) > 5:
            loc_str = f"{locs[0]}, {locs[1]} 외 {len(locs) - 2}곳"
        else:
            loc_str = ", ".join(locs)

        kws = intent.get("keywords", [])
        kws_str = " ".join(kws) if isinstance(kws, list) else str(kws)
        rating_str = f"(★{filters['min_rating']} 이상)" if filters.get("min_rating") else ""
        people_str = f"({filters['people_count']}명 추천)" if filters.get("people_count") else ""

        topic = f"{loc_str} {kws_str} {rating_str} {people_str}".strip() or "요청하신"
        display_name = user_ctx if user_ctx else "회원"

        reply = (
            f"{topic} 방탈출을 추천해드릴게요!\n\n"
            f"맞춤 추천은 **{display_name}**님이 빠방에 작성한 리뷰를 기준으로 가까운 테마를 추천하고\n\n"
            f"조건 추천은 **{display_name}**님이 방금 말씀하신 조건을 필터링해 추천해 드려요!"
        )

        return {
            "reply_text": reply,
            "shown_ids": new_shown,
            "last_filters": filters,
        }

    def route_after_intent(state: BotState) -> str:
        a = state["action"]
        if a == "played_check_inquiry":
            return "inquiry"
        if a in ("played_check", "not_played_check"):
            return "history"
        return "recommend"

    def route_after_aggregate(state: BotState) -> str:
        # rule/personalized/web이 모두 비었을 때만 텍스트 벡터 폴백
        if not (
            state.get("results_rule")
            or state.get("results_personalized")
            or state.get("results_web")
        ):
            return "fallback"
        return "compose"

    return {
        "init_turn": init_turn,
        "analyze_intent": analyze_intent,
        "handle_inquiry": handle_inquiry,
        "handle_history": handle_history,
        "build_filters": build_filters,
        "recommend_rule": recommend_rule,
        "recommend_personalized": recommend_personalized,
        "recommend_web_search": recommend_web_search,
        "aggregate": aggregate,
        "recommend_text_fallback": recommend_text_fallback,
        "compose_reply": compose_reply,
        "route_after_intent": route_after_intent,
        "route_after_aggregate": route_after_aggregate,
    }
