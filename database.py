import os
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

# ==============================================================================
# [라이브러리 안전 로딩 (Safe Import)]
# ==============================================================================
Vector = None
DistanceMeasure = None
FieldFilter = None

# 1. FieldFilter
try:
    from google.cloud.firestore import FieldFilter
except ImportError:
    pass

# 2. Vector
try:
    from google.cloud.firestore import Vector
except ImportError:
    try:
        from google.cloud.firestore_v1.vector import Vector
    except ImportError:
        pass

# 3. DistanceMeasure
try:
    from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
except ImportError:
    pass

# 4. Fallback Classes
if Vector is None:
    class Vector:
        def __init__(self, val): self.value = val

if DistanceMeasure is None:
    class DistanceMeasure:
        COSINE = "COSINE"

# ==============================================================================
# [Firebase 초기화]
# ==============================================================================
@st.cache_resource
def init_firebase():
    """
    Firebase 초기화: Streamlit Secrets를 우선 사용하고, 없으면 로컬 파일을 찾습니다.
    """
    try:
        if not firebase_admin._apps:
            # 1. Streamlit Secrets (배포 환경)
            if "firebase" in st.secrets:
                # st.secrets["firebase"]는 toml 섹션을 dict로 가져옴
                cred_info = dict(st.secrets["firebase"])
                cred = credentials.Certificate(cred_info)
                firebase_admin.initialize_app(cred)
            # 2. 로컬 파일 (개발 환경)
            elif os.path.exists("serviceAccountKey.json"):
                cred = credentials.Certificate("serviceAccountKey.json")
                firebase_admin.initialize_app(cred)
            else:
                return None
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase 초기화 실패: {e}")
        return None


@st.cache_resource(show_spinner="🗂️ 테마 데이터를 캐싱하는 중... (최초 1회)")
def load_themes_snapshot(_db) -> list[dict]:
    """themes 컬렉션을 한 번만 스트리밍해 메모리에 정규화 형태로 캐싱.

    Streamlit 프로세스 수명 동안 유지된다. 새 테마 추가 후 반영하려면 앱 재시작 필요.
    각 항목은 평점/벡터까지 미리 추출되어 있어 호출자는 to_dict 호출이 필요 없다.
    스냅샷은 만족도 평점 내림차순으로 사전 정렬되어 있다 (rule-based 경로 최적화).
    """
    snapshot: list[dict] = []
    for doc in _db.collection("themes").stream():
        data = doc.to_dict() or {}

        vec_obj = data.get("embedding_field")
        vector = None
        try:
            if vec_obj is not None:
                vector = (
                    vec_obj.to_map()["value"]
                    if hasattr(vec_obj, "to_map")
                    else list(vec_obj)
                )
        except Exception:
            vector = None

        snapshot.append(
            {
                "doc_id": doc.id,
                "ref_id": data.get("ref_id"),
                "title": data.get("title") or "",
                "letters": data.get("letters") or "",
                "store_name": data.get("store_name"),
                "location": data.get("location") or "",
                "description": data.get("description") or "",
                "satisfyTotalRating": float(data.get("satisfyTotalRating") or 0),
                "fearTotalRating": float(data.get("fearTotalRating") or 0),
                "activityTotalRating": float(data.get("activityTotalRating") or 0),
                "difficultyTotalRating": float(data.get("difficultyTotalRating") or 0),
                "problemTotalRating": float(data.get("problemTotalRating") or 0),
                "storyTotalRating": float(data.get("storyTotalRating") or 0),
                "interiorTotalRating": float(data.get("interiorTotalRating") or 0),
                "actTotalRating": float(data.get("actTotalRating") or 0),
                "average_person_count": data.get("average_person_count"),
                "vector": vector,
            }
        )

    snapshot.sort(key=lambda t: t["satisfyTotalRating"], reverse=True)
    return snapshot