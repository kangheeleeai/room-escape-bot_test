AREA_GROUPS = [
    {"name": "서울", "keywords": ["서울"], "locations": ["서울", "홍대", "강남", "건대", "대학로", "신촌", "잠실", "신림", "노원", "성수", "영등포", "신사", "수유", "서울대입구", "성신여대", "명동", "천호", "마곡", "용산", "종각", "구로", "목동", "연신내", "동대문", "노량진", "왕십리", "이수", "문래", "역삼"]},
    {"name": "경기/인천", "keywords": ["경기", "인천", "수도권"], "locations": ["인천", "수원", "부천", "성남", "일산", "안산", "의정부", "평택", "동탄", "안양", "김포", "구리", "용인", "화정", "범계", "시흥", "화성", "이천", "하남", "산본", "동두천"]},
    {"name": "충청", "keywords": ["충청", "대전", "세종", "충남", "충북"], "locations": ["대전", "천안", "청주", "당진", "세종"]},
    {"name": "경상", "keywords": ["경상", "부산", "대구", "울산", "경남", "경북"], "locations": ["부산", "대구", "울산", "포항", "창원", "진주", "양산", "구미", "경주", "영주", "안동"]},
    {"name": "전라", "keywords": ["전라", "광주", "전남", "전북"], "locations": ["광주", "전주", "익산", "여수", "목포", "순천", "군산"]},
    {"name": "강원", "keywords": ["강원"], "locations": ["원주", "강릉", "정선", "속초", "춘천"]},
    {"name": "제주", "keywords": ["제주"], "locations": ["제주"]},
]

ALL_LOCATIONS = list({loc for group in AREA_GROUPS for loc in group["locations"]})


def extract_locations(text: str) -> list[str]:
    """텍스트에서 지역명을 추출. 권역 키워드(예: '서울')가 있으면 권역 전체로 확장."""
    found = set()
    text_clean = text.replace(" ", "")

    for loc in ALL_LOCATIONS:
        if loc in text or loc in text_clean:
            found.add(loc)

    for group in AREA_GROUPS:
        for keyword in group["keywords"]:
            if keyword in text or keyword in text_clean:
                found.update(group["locations"])
                break

    return list(found)
