import json
import re
from groq import Groq
from tavily import TavilyClient
from database import firestore, FieldFilter
from utils import sort_candidates_by_query

# ==============================================================================
# [지역 데이터베이스]
# ==============================================================================
AREA_GROUPS = [
    {"name": "서울", "keywords": ["서울"], "locations": ["홍대", "강남", "건대", "대학로", "신촌", "잠실", "신림", "노원", "성수", "영등포", "신사", "수유", "서울대입구", "성신여대", "명동", "천호", "마곡", "용산", "종각", "구로", "목동", "연신내", "동대문", "노량진", "왕십리", "이수", "문래", "역삼"]},
    {"name": "경기/인천", "keywords": ["경기", "인천", "수도권"], "locations": ["인천", "수원", "부천", "성남", "일산", "안산", "의정부", "평택", "동탄", "안양", "김포", "구리", "용인", "화정", "범계", "시흥", "화성", "이천", "하남", "산본", "동두천"]},
    {"name": "충청", "keywords": ["충청", "대전", "세종", "충남", "충북"], "locations": ["대전", "천안", "청주", "당진", "세종"]},
    {"name": "경상", "keywords": ["경상", "부산", "대구", "울산", "경남", "경북"], "locations": ["부산", "대구", "울산", "포항", "창원", "진주", "양산", "구미", "경주", "영주", "안동"]},
    {"name": "전라", "keywords": ["전라", "광주", "전남", "전북"], "locations": ["광주", "전주", "익산", "여수", "목포", "순천", "군산"]},
    {"name": "강원", "keywords": ["강원"], "locations": ["원주", "강릉", "정선", "속초", "춘천"]}, # 순천 중복 제거 및 주요 도시 보완
    {"name": "제주", "keywords": ["제주"], "locations": ["제주"]}
]

# 검색 속도를 위한 전체 지역 평탄화 리스트
ALL_LOCATIONS = []
for group in AREA_GROUPS:
    ALL_LOCATIONS.extend(group['locations'])
ALL_LOCATIONS = list(set(ALL_LOCATIONS)) # 중복 제거

class EscapeBotEngine:
    def __init__(self, vector_recommender, rule_recommender, groq_key, tavily_key):
        self.vector_recommender = vector_recommender
        self.rule_recommender = rule_recommender 
        self.db = rule_recommender.db
        
        self.tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None
        
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
            self.model_name = "llama-3.3-70b-versatile"
        else:
            self.groq_client = None

    def _clean_json_string(self, json_str):
        if not json_str: return ""
        cleaned = re.sub(r"```json\s*", "", json_str)
        cleaned = re.sub(r"```", "", cleaned)
        return cleaned.strip()

    def _call_llm(self, prompt, json_mode=False):
        if not self.groq_client: return None
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for Escape Room recommendations. Always respond in Korean." + (" Output JSON only." if json_mode else "")
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=0.1, 
                response_format={"type": "json_object"} if json_mode else None,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return None

    def _extract_locations_from_text(self, text, on_log=None):
        """
        정해진 지역 리스트를 기반으로 텍스트에서 지역명을 추출합니다.
        대분류(예: 경상)가 발견되면 해당 권역의 모든 소분류를 추가합니다.
        """
        found_locations = set()
        text_clean = text.replace(" ", "") # 띄어쓰기 무시 매칭을 위해

        # 1. 소분류 (개별 지역) 매칭
        # "홍대" -> found
        for loc in ALL_LOCATIONS:
            if loc in text or loc in text_clean:
                found_locations.add(loc)

        # 2. 대분류 (권역) 매칭 및 확장
        # "경상도" -> 부산, 대구, 울산... 전부 추가
        for group in AREA_GROUPS:
            for keyword in group['keywords']:
                if keyword in text or keyword in text_clean:
                    # 키워드가 발견되었는데, 이미 해당 그룹의 소분류가 하나도 발견되지 않았거나
                    # 혹은 광역 검색 의도("경상도 방탈출")일 경우를 위해 확장
                    # (여기서는 단순히 키워드가 있으면 해당 그룹 전체를 후보에 넣습니다. OR 검색이므로 안전함)
                    if on_log: on_log(f"   -> 권역 감지: '{keyword}' ({len(group['locations'])}개 지역 추가)")
                    found_locations.update(group['locations'])
                    break # 한 그룹에서 키워드 하나만 걸리면 됨

        return list(found_locations)

    def find_theme_id(self, location, theme_name, on_log=None):
        if on_log: on_log(f"[DB] 테마 검색: {theme_name} (지역: {location})")
        
        try:
            themes_ref = self.db.collection('themes')
            query = themes_ref
            if location:
                query = query.where(filter=FieldFilter("location", "==", location))
            
            docs = list(query.limit(200).stream())
            target_name = theme_name.replace(" ", "")
            
            for doc in docs:
                data = doc.to_dict()
                title = data.get('title', '')
                letters = data.get('letters', '')
                if target_name in title.replace(" ", ""):
                    if on_log: on_log(f"   -> 발견: {title} (ID: {doc.id})")
                    return int(data.get('ref_id') or doc.id)
                if letters and target_name in letters.replace(" ", ""):
                    return int(data.get('ref_id') or doc.id)
            
            if on_log: on_log("   -> 검색 실패")
            return None
        except Exception as e:
            if on_log: on_log(f"   ⚠️ 검색 에러: {e}")
            return None

    def update_play_history(self, nickname, theme_id, action, on_log=None):
        try:
            users_ref = self.db.collection('users')
            q = users_ref.where(filter=FieldFilter("nickname", "==", nickname)).limit(1)
            docs = list(q.stream())
            
            if not docs: return "❌ 유저 미등록"
            
            user_doc = docs[0]
            if action == "played_check":
                user_doc.reference.update({"played": firestore.ArrayUnion([theme_id])})
                if on_log: on_log(f"[기록] {nickname}님 플레이 리스트에 {theme_id} 추가")
                return "추가 완료"
            elif action == "not_played_check":
                user_doc.reference.update({"played": firestore.ArrayRemove([theme_id])})
                if on_log: on_log(f"[기록] {nickname}님 플레이 리스트에서 {theme_id} 삭제")
                return "삭제 완료"
            return "알 수 없는 요청"
        except Exception as e:
            return f"에러: {e}"

    def analyze_user_intent(self, user_query, on_log=None):
        if on_log: on_log(f"[LLM] 사용자 의도 분석 중... ('{user_query}')")
        
        if not self.groq_client: return {}
        
        # [변경] LLM에게 지역 추출을 시키지 않음 (파이썬 로직으로 처리)
        prompt = f"""
        Analyze the user's Escape Room query in Korean.
        Query: "{user_query}"
        
        Rules:
        1. "played_check_inquiry": Asking how to record played themes.
        2. "played_check": User says they played a theme (e.g. "강남 링 했어").
        3. "not_played_check": User wants to cancel a record.
        4. "recommend": User asks for recommendations.
        5. "another_recommend": Asking for other options.
        
        Extract:
        - action (string)
        - keywords (Array of strings: genres, vibes, etc. EXCLUDING location names)
        - mentioned_users (Array of strings)
        - items (Array of objects {{theme: "theme_name", location: "loc_name"}} only for played_check)
        
        Return JSON only.
        """
        try:
            result_str = self._call_llm(prompt, json_mode=True)
            cleaned_str = self._clean_json_string(result_str)
            result = json.loads(cleaned_str)
            
            # [핵심] 지역 추출은 파이썬 로직으로 수행하여 덮어쓰기
            extracted_locs = self._extract_locations_from_text(user_query, on_log)
            result['locations'] = extracted_locs
            
            # played_check일 경우 items 내부의 location도 보정 시도 (단순 매칭)
            if result.get('items'):
                for item in result['items']:
                    if not item.get('location') and extracted_locs:
                        item['location'] = extracted_locs[0] # 첫 번째 발견된 지역 할당

            if on_log: on_log(f"   -> 분석 완료: {result.get('action')}, 추출 지역: {result.get('locations')}")
            return result
        except Exception as e:
            if on_log: on_log(f"   ❌ 의도 분석 실패: {e}")
            # 에러 시에도 지역 추출은 시도
            locs = self._extract_locations_from_text(user_query)
            return {"action": "recommend", "keywords": [user_query], "locations": locs}

    def generate_reply(self, user_query, user_context=None, session_context=None, on_log=None):
        if not self.groq_client:
            return "⚠️ API Key 설정 필요", {}, {}, "error", {}

        # 1. 의도 분석 (여기서 파이썬 지역 추출이 실행됨)
        intent_data = self.analyze_user_intent(user_query, on_log)
        action = intent_data.get('action', 'recommend')
        debug_info = {"intent": intent_data, "query": user_query}

        # 플레이 기록 문의
        if action == "played_check_inquiry":
            msg = "플레이한 테마를 `[지역] [테마명] 했어` 라고 말씀해주시면 기록해 드립니다!"
            return msg, {}, {}, action, debug_info

        # 플레이 기록 추가/삭제
        if action in ['played_check', 'not_played_check']:
            if not user_context:
                return "⚠️ 닉네임을 먼저 설정해주세요.", {}, {}, action, debug_info
            
            items = intent_data.get('items') or []
            # Fallback for single item
            if not items and intent_data.get('theme'):
                loc = intent_data.get('locations')[0] if intent_data.get('locations') else ""
                items.append({"location": loc, "theme": intent_data.get('theme')})

            results_msg = []
            
            for item in items:
                loc = item.get('location')
                theme = item.get('theme')
                if theme:
                    tid = self.find_theme_id(loc, theme, on_log)
                    if tid:
                        res = self.update_play_history(user_context, tid, action, on_log)
                        results_msg.append(f"- **{theme}**: {res}")
                    else:
                        results_msg.append(f"- **{theme}**: ⚠️ 테마 못 찾음")
            
            return "\n".join(results_msg), {}, {}, action, debug_info

        # 3. 필터 설정 (locations 리스트 사용)
        current_filters = {
            'locations': intent_data.get('locations') or [],
            'keywords': intent_data.get('keywords') or [],
            'mentioned_users': intent_data.get('mentioned_users') or []
        }
        
        # 유저 처리
        current_users = [u.strip() for u in str(user_context or "").split(',') if u.strip()]
        for u in current_filters['mentioned_users']:
            if u not in current_users: current_users.append(u)
        
        final_context = current_users if len(current_users) > 1 else (current_users[0] if current_users else None)

        filters_to_use = {}
        exclude_ids = []
        
        if action == 'another_recommend' and session_context:
            filters_to_use = session_context.get('last_filters', {})
            exclude_ids = list(session_context.get('shown_ids', []))
            # 위치 변경 요청이 있으면 덮어쓰기 (새로운 지역이 감지되었을 때만)
            if current_filters.get('locations'): 
                filters_to_use['locations'] = current_filters['locations']
        else:
            filters_to_use = current_filters
            exclude_ids = []

        if on_log: on_log(f"필터 적용: {filters_to_use}, 제외 ID: {len(exclude_ids)}개")

        # 4. 추천 실행
        final_results = {}
        
        # Rule-Based
        candidates_rule = self.rule_recommender.search_themes(
            filters_to_use, user_query, limit=3, nicknames=final_context, exclude_ids=exclude_ids, log_func=on_log
        )
        if candidates_rule: final_results['rule_based'] = candidates_rule

        # Personalized
        if final_context:
            candidates_vector = self.vector_recommender.recommend_by_user_search(
                final_context, user_query=user_query, limit=3, filters=filters_to_use, exclude_ids=exclude_ids, log_func=on_log
            )
            if candidates_vector:
                final_results['personalized'] = candidates_vector

        # Fallback
        if not final_results:
            candidates_text = self.vector_recommender.recommend_by_text(
                user_query, filters=filters_to_use, exclude_ids=exclude_ids, log_func=on_log
            )
            if candidates_text:
                final_results['text_search'] = sort_candidates_by_query(candidates_text, user_query)[:3]
            else:
                return "조건에 맞는 테마를 찾지 못했습니다.", {}, filters_to_use, action, debug_info

        # 5. 답변 생성
        if on_log: on_log("📝 답변 생성 중 (Fixed Template)...")
        
        # 토픽 문자열 구성
        locs = intent_data.get('locations') or []
        # 너무 많으면 축약 (예: 경상도 검색 시)
        if len(locs) > 5:
            loc_str = f"{locs[0]}, {locs[1]} 외 {len(locs)-2}곳"
        else:
            loc_str = ", ".join(locs) if locs else ""
        
        keywords = intent_data.get('keywords', [])
        kws_str = " ".join(keywords) if isinstance(keywords, list) else str(keywords)
        
        topic_str = f"{loc_str} {kws_str}".strip()
        if not topic_str: topic_str = "요청하신"

        display_name = user_context if user_context else "회원"

        response_text = f"{topic_str} 방탈출을 추천해드릴게요!\n\n" \
                        f"맞춤 추천은 **{display_name}**님이 빠방에 작성한 리뷰를 기준으로 가까운 테마를 추천하고\n" \
                        f"조건 추천은 **{display_name}**님이 방금 말씀하신 조건을 필터링해 추천해 드려요!"

        return response_text, final_results, filters_to_use, action, debug_info
