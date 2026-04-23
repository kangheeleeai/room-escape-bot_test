"""utils.sort_candidates_by_query가 인식해서 정렬에 활용할 수 있는 키워드 어휘.

이 어휘에 단 하나도 매칭되지 않는 사용자 입력은 'rule-based 정렬을 의미있게 적용할 수 없음'으로
간주하고 웹 검색 보강 추천(recommend_web_search)을 트리거한다.
"""

RECOGNIZED_KEYWORDS: tuple[str, ...] = (
    # 공포 / 비공포
    "공포", "무서운", "호러", "스릴러",
    "안무서운", "무섭지 않은", "겁쟁이", "극쫄",
    # 난이도 / 문제방
    "쉬운", "안어려운", "입문", "초보",
    "문제방", "어려운", "문제", "숙련자",
    # 활동성
    "활동", "동적인", "바지", "체력",
    "활동적이지 않은", "치마", "힐", "걷는",
    # 스토리 / 인테리어 / 연출
    "스토리", "드라마", "감성", "서사",
    "인테리어", "리얼리티", "실제같은", "배경",
    "연출", "장치", "화려", "스케일",
)


def has_recognized_keyword(query: str) -> bool:
    """쿼리 내에 정렬에 활용 가능한 키워드가 하나라도 존재하면 True."""
    if not query:
        return False
    return any(kw in query for kw in RECOGNIZED_KEYWORDS)
