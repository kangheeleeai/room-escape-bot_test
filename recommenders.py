import numpy as np

from database import FieldFilter, load_themes_snapshot
from utils import sort_candidates_by_query


def _normalize_users(nicknames) -> list[str]:
    if isinstance(nicknames, str):
        return [n.strip() for n in nicknames.split(",") if n.strip()]
    if isinstance(nicknames, list):
        return [n.strip() for n in nicknames if n and str(n).strip()]
    return []


def _id_excluded(theme: dict, exclude_ids: set) -> bool:
    """exclude_ids는 int / str / doc_id 혼재 가능 → 모두 검사."""
    try:
        tid = int(theme["ref_id"] or theme["doc_id"])
        if tid in exclude_ids or str(tid) in exclude_ids:
            return True
    except Exception:
        pass
    return theme["doc_id"] in exclude_ids


def _location_match(theme: dict, clean_locs: list[str]) -> bool:
    if not clean_locs:
        return True
    db_loc = theme["location"].replace(" ", "")
    return any(target in db_loc for target in clean_locs)


def _people_match(theme: dict, target: int) -> bool:
    """원본과 동일한 ±1 오차 범위 매칭. average_person_count 미지정 시 통과."""
    avg = theme.get("average_person_count")
    if not avg:
        return True
    try:
        rec_val = float(avg)
        return float(target) - 1 <= rec_val <= float(target) + 1
    except Exception:
        return True


def _to_card(theme: dict, *, with_vector: bool = False, score: float | None = None) -> dict:
    card = {
        "id": theme["doc_id"],
        "title": theme["title"],
        "store": theme["store_name"],
        "location": theme["location"],
        "desc": theme["description"][:150],
        "rating": theme["satisfyTotalRating"],
        "fear": theme["fearTotalRating"],
        "activity": theme["activityTotalRating"],
        "difficulty": theme["difficultyTotalRating"],
        "problem": theme["problemTotalRating"],
        "story": theme["storyTotalRating"],
        "interior": theme["interiorTotalRating"],
        "act": theme["actTotalRating"],
    }
    if with_vector:
        card["vector"] = theme["vector"]
    if score is not None:
        card["score"] = score
    return card


class RuleBasedRecommender:
    def __init__(self, db):
        self.db = db

    def search_themes(
        self,
        criteria,
        user_query="",
        limit=30,
        nicknames=None,
        exclude_ids=None,
        log_func=None,
    ):
        locs_input = criteria.get("locations", []) or []
        min_rating = criteria.get("min_rating")
        people_count = criteria.get("people_count")

        log_parts = [f"지역: {', '.join(locs_input) if locs_input else '전체'}"]
        if min_rating:
            log_parts.append(f"최소평점 {min_rating}")
        if people_count:
            log_parts.append(f"인원 {people_count}명")
        if log_func:
            log_func(f"[Rule] 검색 시작 ({', '.join(log_parts)})")

        # 1. 플레이 기록 조회 (users 컬렉션은 캐싱 대상 아님)
        target_users = _normalize_users(nicknames)
        played_theme_ids: set[int] = set()
        if target_users:
            try:
                users_ref = self.db.collection("users")
                if len(target_users) > 10:
                    target_users = target_users[:10]
                user_q = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
                for u_doc in user_q.stream():
                    for pid in u_doc.to_dict().get("played", []) or []:
                        played_theme_ids.add(int(pid))
                if log_func:
                    log_func(
                        f"   -> {len(target_users)}명 플레이 기록 {len(played_theme_ids)}개 제외"
                    )
            except Exception as e:
                if log_func:
                    log_func(f"   ⚠️ 이력 조회 에러: {e}")

        total_exclude_ids: set = set(exclude_ids or [])
        total_exclude_ids.update(played_theme_ids)

        clean_locs = [loc.replace(" ", "") for loc in locs_input if loc.strip()]
        min_rating_f = float(min_rating) if min_rating else None

        # 2. 캐시된 스냅샷 순회 (사전 정렬: 평점 내림차순)
        snapshot = load_themes_snapshot(self.db)
        raw_candidates: list[dict] = []
        for theme in snapshot:
            # 평점 사전정렬 → 컷오프 도달 시 조기 종료
            if min_rating_f is not None and theme["satisfyTotalRating"] < min_rating_f:
                break
            if _id_excluded(theme, total_exclude_ids):
                continue
            if not _location_match(theme, clean_locs):
                continue
            if people_count and not _people_match(theme, people_count):
                continue
            raw_candidates.append(_to_card(theme, with_vector=True))

        sorted_candidates = sort_candidates_by_query(raw_candidates, user_query)
        if log_func:
            log_func(f"   -> [Rule] 필터링 후 {len(sorted_candidates)}개 후보 발견")
        return sorted_candidates[:limit]


class VectorRecommender:
    def __init__(self, db, model):
        self.db = db
        self.model = model

    def get_group_vector(self, nicknames, log_func=None):
        target_users = _normalize_users(nicknames)
        if not target_users:
            return None
        try:
            users_ref = self.db.collection("users")
            if len(target_users) > 10:
                target_users = target_users[:10]
            docs = list(
                users_ref.where(filter=FieldFilter("nickname", "in", target_users)).stream()
            )

            vectors = []
            for doc in docs:
                vec_obj = doc.to_dict().get("embedding_field")
                if vec_obj is None:
                    continue
                try:
                    v = (
                        vec_obj.to_map()["value"]
                        if hasattr(vec_obj, "to_map")
                        else list(vec_obj)
                    )
                    vectors.append(v)
                except Exception:
                    pass

            if not vectors:
                return None

            mean_vector = np.mean(np.array(vectors), axis=0)
            norm = np.linalg.norm(mean_vector)
            return (mean_vector / norm).tolist() if norm > 0 else mean_vector.tolist()

        except Exception as e:
            if log_func:
                log_func(f"   ⚠️ 벡터 계산 실패: {e}")
            return None

    def _get_played_ids_internal(self, user_context, log_func=None):
        target_users = _normalize_users(user_context)
        played_ids: set[int] = set()
        if not target_users:
            return played_ids
        try:
            users_ref = self.db.collection("users")
            if len(target_users) > 10:
                target_users = target_users[:10]
            user_q = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
            for doc in user_q.stream():
                for pid in doc.to_dict().get("played", []) or []:
                    played_ids.add(int(pid))
            if log_func:
                log_func(
                    f"   -> [Vector] {len(target_users)}명 이력 {len(played_ids)}개 로드"
                )
        except Exception as e:
            if log_func:
                log_func(f"   ⚠️ [Vector] 이력 조회 에러: {e}")
        return played_ids

    def _execute_vector_search(
        self, vector, limit=20, filters=None, exclude_ids=None, log_func=None
    ):
        try:
            filters = filters or {}
            locs_input = filters.get("locations", []) or []
            min_rating = filters.get("min_rating")
            people_count = filters.get("people_count")

            clean_locs = [loc.replace(" ", "") for loc in locs_input if loc.strip()]
            min_rating_f = float(min_rating) if min_rating else None
            total_exclude_ids: set = set(exclude_ids or [])

            target_vec = np.array(vector)
            target_norm = float(np.linalg.norm(target_vec))

            snapshot = load_themes_snapshot(self.db)
            candidates: list[dict] = []

            for theme in snapshot:
                if _id_excluded(theme, total_exclude_ids):
                    continue
                if not _location_match(theme, clean_locs):
                    continue
                if min_rating_f is not None and theme["satisfyTotalRating"] < min_rating_f:
                    continue
                if people_count and not _people_match(theme, people_count):
                    continue

                theme_vec = theme["vector"]
                if not theme_vec:
                    continue

                try:
                    tv = np.array(theme_vec)
                    theme_norm = float(np.linalg.norm(tv))
                    if target_norm > 0 and theme_norm > 0:
                        score = float(np.dot(target_vec, tv) / (target_norm * theme_norm))
                    else:
                        score = 0.0
                except Exception:
                    score = 0.0

                candidates.append(_to_card(theme, score=score))

            candidates.sort(key=lambda x: x["score"], reverse=True)
            if log_func:
                log_func(f"   -> [Vector] {len(candidates)}개 후보 중 Top {limit} 추출")
            return candidates[:limit]

        except Exception as e:
            if log_func:
                log_func(f"   ❌ [Error] Vector Search 실패: {e}")
            return []

    def recommend_by_text(self, query_text, filters=None, exclude_ids=None, log_func=None):
        if not self.model:
            return []
        if log_func:
            log_func(f"[Text] '{query_text}' 임베딩 검색 시작")
        query_vector = self.model.encode(query_text).tolist()
        return self._execute_vector_search(
            query_vector, limit=10, filters=filters, exclude_ids=exclude_ids, log_func=log_func
        )

    def recommend_by_user_search(
        self, user_context, user_query="", limit=3, filters=None, exclude_ids=None, log_func=None
    ):
        if log_func:
            log_func(f"[Person] '{user_context}' 벡터 분석 (키워드: '{user_query}')")

        target_vec = self.get_group_vector(user_context, log_func)
        if not target_vec:
            return []

        played_ids = self._get_played_ids_internal(user_context, log_func)
        final_exclude = set(exclude_ids or [])
        final_exclude.update(played_ids)

        fetch_limit = limit * 5 if user_query else limit
        candidates = self._execute_vector_search(
            target_vec, limit=fetch_limit, filters=filters, exclude_ids=final_exclude, log_func=log_func
        )

        if user_query and candidates:
            candidates = sort_candidates_by_query(candidates, user_query)
            if log_func:
                log_func(f"   -> [Re-rank] 키워드('{user_query}') 반영하여 재정렬 완료")

        return candidates[:limit]
