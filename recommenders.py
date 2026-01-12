import numpy as np
from database import firestore, Vector, DistanceMeasure, FieldFilter
from utils import sort_candidates_by_query
from config import PROJECT_ID

class RuleBasedRecommender:
    def __init__(self, db):
        self.db = db

    def search_themes(self, criteria, user_query="", limit=30, nicknames=None, exclude_ids=None, log_func=None):
        locs_input = criteria.get('locations', [])
        loc_str_log = ", ".join(locs_input) if locs_input else "전체"
        
        min_rating = criteria.get('min_rating')
        # 인원수 필터 로그 추가
        people_count = criteria.get('people_count')
        
        log_parts = [f"지역: {loc_str_log}"]
        if min_rating: log_parts.append(f"최소평점 {min_rating}")
        if people_count: log_parts.append(f"인원 {people_count}명")
        
        if log_func: log_func(f"[Rule] 검색 시작 ({', '.join(log_parts)})")
        
        played_theme_ids = set()
        
        # 1. 이력 조회
        target_users = []
        if isinstance(nicknames, str):
            target_users = [n.strip() for n in nicknames.split(',') if n.strip()]
        elif isinstance(nicknames, list):
            target_users = nicknames

        if target_users:
            try:
                users_ref = self.db.collection('users')
                if len(target_users) > 10: target_users = target_users[:10]
                user_q = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
                user_docs = list(user_q.stream())
                
                for u_doc in user_docs:
                    played = u_doc.to_dict().get('played', [])
                    for pid in played: played_theme_ids.add(int(pid))
                
                if log_func: log_func(f"   -> {len(target_users)}명 플레이 기록 {len(played_theme_ids)}개 제외")
            except Exception as e:
                if log_func: log_func(f"   ⚠️ 이력 조회 에러: {e}")

        total_exclude_ids = set(exclude_ids) if exclude_ids else set()
        total_exclude_ids.update(played_theme_ids)

        # 2. DB 쿼리
        themes_ref = self.db.collection('themes')
        query = themes_ref.order_by('satisfyTotalRating', direction="DESCENDING").limit(200)

        docs = list(query.stream())
        raw_candidates = []
        
        clean_locs = [loc.replace(" ", "") for loc in locs_input if loc.strip()]

        for doc in docs:
            data = doc.to_dict()
            
            # ID 제외
            try:
                tid = int(data.get('ref_id') or doc.id)
                if tid in total_exclude_ids or str(tid) in total_exclude_ids: continue
            except: 
                if doc.id in total_exclude_ids: continue

            # [지역 필터링]
            if clean_locs:
                db_loc = data.get('location', '').replace(" ", "")
                if not any(target in db_loc for target in clean_locs):
                    continue

            # [평점 필터링]
            try:
                rating = float(data.get('satisfyTotalRating') or 0)
                if min_rating and rating < float(min_rating):
                    continue
            except:
                pass

            # [인원수 필터링]
            # average_person_count 필드를 확인하여 N-1 ~ N+1 범위 내인지 검사
            if people_count:
                try:
                    avg_person = data.get('average_person_count')
                    if avg_person: # 데이터가 있는 경우만 필터링
                        rec_val = float(avg_person)
                        target = float(people_count)
                        # 오차 범위 ±1 (예: 4명 요청 시 3~5명 추천)
                        if not (target - 1 <= rec_val <= target + 1):
                            continue
                except:
                    pass

            # 벡터 저장
            vec_obj = data.get('embedding_field')
            vector = None
            try:
                if vec_obj: vector = vec_obj.to_map()['value'] if hasattr(vec_obj, 'to_map') else list(vec_obj)
            except: pass

            raw_candidates.append({
                'id': doc.id,
                'title': data.get('title'),
                'store': data.get('store_name'),
                'location': data.get('location'),
                'desc': data.get('description', '')[:150],
                'rating': float(data.get('satisfyTotalRating') or 0),
                'fear': float(data.get('fearTotalRating') or 0),
                'activity': float(data.get('activityTotalRating') or 0),
                'difficulty': float(data.get('difficultyTotalRating') or 0),
                'vector': vector
            })

        sorted_candidates = sort_candidates_by_query(raw_candidates, user_query)
        if log_func: log_func(f"   -> [Rule] 필터링 후 {len(sorted_candidates)}개 후보 발견")
        
        return sorted_candidates[:limit]


class VectorRecommender:
    def __init__(self, db, model):
        self.db = db
        self.model = model

    def get_group_vector(self, nicknames, log_func=None):
        target_users = [n.strip() for n in nicknames if n.strip()] if isinstance(nicknames, list) else ([n.strip() for n in nicknames.split(',')] if nicknames else [])
        if not target_users: return None

        try:
            users_ref = self.db.collection('users')
            if len(target_users) > 10: target_users = target_users[:10]
            docs = list(users_ref.where(filter=FieldFilter("nickname", "in", target_users)).stream())
            
            vectors = []
            for doc in docs:
                vec_obj = doc.to_dict().get('embedding_field')
                if vec_obj:
                    try:
                        v = vec_obj.to_map()['value'] if hasattr(vec_obj, 'to_map') else list(vec_obj)
                        vectors.append(v)
                    except: pass
            
            if not vectors: return None
            
            mean_vector = np.mean(np.array(vectors), axis=0)
            norm = np.linalg.norm(mean_vector)
            return (mean_vector / norm).tolist() if norm > 0 else mean_vector.tolist()

        except Exception as e:
            if log_func: log_func(f"   ⚠️ 벡터 계산 실패: {e}")
            return None

    def _get_played_ids_internal(self, user_context, log_func=None):
        played_ids = set()
        if not user_context: return played_ids
        
        target_users = []
        if isinstance(user_context, str):
            target_users = [n.strip() for n in user_context.split(',') if n.strip()]
        elif isinstance(user_context, list):
            target_users = user_context
            
        if not target_users: return played_ids

        try:
            users_ref = self.db.collection('users')
            if len(target_users) > 10: target_users = target_users[:10]
            user_q = users_ref.where(filter=FieldFilter("nickname", "in", target_users))
            docs = user_q.stream()
            for doc in docs:
                played = doc.to_dict().get('played', [])
                for pid in played:
                    played_ids.add(int(pid))
            if log_func: log_func(f"   -> [Vector] {len(target_users)}명 이력 {len(played_ids)}개 로드")
        except Exception as e:
            if log_func: log_func(f"   ⚠️ [Vector] 이력 조회 에러: {e}")
            
        return played_ids

    def _execute_vector_search(self, vector, limit=20, filters=None, exclude_ids=None, log_func=None):
        try:
            themes_ref = self.db.collection('themes')
            query = themes_ref
            
            locs_input = filters.get('locations', []) if filters else []
            min_rating = filters.get('min_rating') if filters else None
            # 인원수 필터 추가
            people_count = filters.get('people_count') if filters else None
            
            clean_locs = [loc.replace(" ", "") for loc in locs_input if loc.strip()]
            
            # 제한 없이 전체 로드 (메모리 필터링)
            docs = list(query.stream())
            
            candidates = []
            target_vec = np.array(vector)
            target_norm = np.linalg.norm(target_vec)
            
            total_exclude_ids = set(exclude_ids) if exclude_ids else set()

            for doc in docs:
                data = doc.to_dict()
                
                # 제외 ID 필터링
                try:
                    tid = int(data.get('ref_id') or doc.id)
                    if tid in total_exclude_ids or str(tid) in total_exclude_ids: continue
                except:
                    if doc.id in total_exclude_ids: continue

                # [지역 필터]
                if clean_locs:
                    db_loc = data.get('location', '').replace(" ", "")
                    if not any(target in db_loc for target in clean_locs):
                        continue

                # [평점 필터]
                try:
                    rating = float(data.get('satisfyTotalRating') or 0)
                    if min_rating and rating < float(min_rating):
                        continue
                except:
                    pass

                # [인원수 필터]
                if people_count:
                    try:
                        avg_person = data.get('average_person_count')
                        if avg_person:
                            rec_val = float(avg_person)
                            target = float(people_count)
                            if not (target - 1 <= rec_val <= target + 1):
                                continue
                    except:
                        pass
                
                # 벡터 유사도 계산
                vec_obj = data.get('embedding_field')
                if not vec_obj: continue
                
                try:
                    theme_vec = vec_obj.to_map()['value'] if hasattr(vec_obj, 'to_map') else list(vec_obj)
                    theme_vec = np.array(theme_vec)
                    theme_norm = np.linalg.norm(theme_vec)
                    
                    if target_norm > 0 and theme_norm > 0:
                        score = np.dot(target_vec, theme_vec) / (target_norm * theme_norm)
                    else:
                        score = 0
                except:
                    score = 0
                
                candidates.append({
                    'id': doc.id,
                    'title': data.get('title'),
                    'store': data.get('store_name'),
                    'location': data.get('location'),
                    'desc': data.get('description', '')[:150],
                    'rating': float(data.get('satisfyTotalRating') or 0),
                    'fear': float(data.get('fearTotalRating') or 0),
                    'difficulty': float(data.get('difficultyTotalRating') or 0),
                    'activity': float(data.get('activityTotalRating') or 0),
                    'problem': float(data.get('problemTotalRating') or 0),
                    'story': float(data.get('storyTotalRating') or 0),
                    'interior': float(data.get('interiorTotalRating') or 0),
                    'act': float(data.get('actTotalRating') or 0),
                    'score': score
                })

            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            if log_func: log_func(f"   -> [Vector] {len(candidates)}개 후보 중 Top {limit} 추출")
            return candidates[:limit]

        except Exception as e:
            if log_func: log_func(f"   ❌ [Error] Vector Search 실패: {e}")
            return []

    def recommend_by_text(self, query_text, filters=None, exclude_ids=None, log_func=None):
        if not self.model: return []
        if log_func: log_func(f"[Text] '{query_text}' 임베딩 검색 시작")
        query_vector = self.model.encode(query_text).tolist()
        return self._execute_vector_search(query_vector, limit=10, filters=filters, exclude_ids=exclude_ids, log_func=log_func)

    def recommend_by_user_search(self, user_context, user_query="", limit=3, filters=None, exclude_ids=None, log_func=None):
        if log_func: log_func(f"[Person] '{user_context}' 벡터 분석 (키워드: '{user_query}')")
        
        target_vec = self.get_group_vector(user_context, log_func)
        if not target_vec: return []
        
        played_ids = self._get_played_ids_internal(user_context, log_func)
        final_exclude = set(exclude_ids) if exclude_ids else set()
        final_exclude.update(played_ids)
        
        fetch_limit = limit * 5 if user_query else limit
        
        candidates = self._execute_vector_search(target_vec, limit=fetch_limit, filters=filters, exclude_ids=final_exclude, log_func=log_func)
        
        if user_query and candidates:
            candidates = sort_candidates_by_query(candidates, user_query)
            if log_func: log_func(f"   -> [Re-rank] 키워드('{user_query}') 반영하여 재정렬 완료")
            
        return candidates[:limit]
