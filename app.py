import logging
import uuid

import streamlit as st

from config import GROQ_API_KEY, TAVILY_API_KEY
from database import init_firebase
from graph import compile_bot_graph
from models import load_embed_model
from recommenders import RuleBasedRecommender, VectorRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="방탈출 AI (LangGraph)", page_icon="🕵️", layout="wide")

st.markdown(
    """
<style>
    .theme-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    .theme-title { font-weight: bold; font-size: 1.1em; color: #000000 !important; }
    .theme-meta  { font-size: 0.9em; color: #555; }
    .theme-desc  { font-size: 0.9em; margin-top: 5px; color: #333; }
</style>
""",
    unsafe_allow_html=True,
)

GUIDE_MD = """
## 🕵️ 방탈출 AI 사용 설명서

### 1️⃣ 기본 추천
* "강남에서 공포 테마 추천해줘"
* "홍대에서 활동성 많은거"
* "다른거 추천해줘" -> 지금 추천 받은 테마를 제외하고 동일한 조건의 다른 테마를 추천합니다.

### 2️⃣ 닉네임 맞춤 추천
* 왼쪽 사이드바에 닉네임을 입력하면 **내 플레이 기록**을 제외하고 추천합니다.
* 친구들과 함께라면 쉼표(`,`)로 여러 명을 입력하세요.

### 3️⃣ 기록 관리
* "**강남에 있는 링 했어**" -> 플레이 목록에 추가
* "**홍대에 있는 삐릿뽀 안했어**" -> 기록 취소

### 0️⃣ 주의
* 가능한 키워드: 공포(무서운, 안무서운 등), 연출, 인테리어, 스토리, 인원 관련, 문제방(어려운, 문제방, 안어려운 등), 활동성
* 불가능 키워드: 판타지, SF, 코미디, 코믹, 서브 여부
* 데이터 수집의 문제로 오래전에 빠방에 등록되었던 리뷰는 수집하지 못했습니다. 기록관리 기능으로 입력해두시면 영구적으로 기록됩니다.
* 긴 요청사항은 AI가 이해하지 못합니다. 간결하게 작성하고 "다른거 추천해줘" 기능을 사용해주세요.
* 에러 발생, 원치않는 결과가 나올시 "대화 지우기", "새로 고침" 후 재입력 부탁드립니다.
"""


def render_cards(card_list):
    if not card_list:
        st.caption("결과가 없습니다.")
        return

    for item in card_list:
        desc = item.get("desc", "")
        rating = item.get("rating", 0.0)
        if len(desc) > 100:
            desc = desc[:100] + "..."

        st.markdown(
            f"""
        <div class='theme-card'>
            <div class='theme-title'>{item['title']} <span style='font-size:0.8em; color:black'>({item['store']})</span></div>
            <div class='theme-meta'>⭐ 평점: {rating:.2f} | 📍 {item['location']}</div>
            <hr style="margin: 8px 0; opacity: 0.2;">
            <div class='theme-desc' style='white-space: pre-wrap; line-height: 1.5;'>{desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_assistant_message(msg: dict):
    st.markdown(msg["content"])
    cards = msg.get("cards") or {}
    if not cards:
        return

    tab1, tab2 = st.tabs(["🔎 조건 추천", "🎯 맞춤 추천"])
    with tab1:
        rule_list = cards.get("rule_based") or []
        web_list = cards.get("web_search") or []
        if not rule_list and cards.get("text_search"):
            st.info("조건에 딱 맞는 테마가 없어 유사한 테마를 보여드립니다.")
            rule_list = cards["text_search"]
        if rule_list:
            render_cards(rule_list)
        if web_list:
            if rule_list:
                st.markdown("---")
            st.info("🌐 인식되지 않은 키워드는 웹 검색으로 보강 추천드려요.")
            render_cards(web_list)
        if not rule_list and not web_list:
            st.caption("검색 결과가 없습니다.")
    with tab2:
        if cards.get("personalized"):
            render_cards(cards["personalized"])
        else:
            st.caption("맞춤 추천 결과가 없습니다. (로그인 필요)")


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "어떤 방탈출 테마를 찾으시나요? 지역이나 장르를 말씀해주세요!"},
            {"role": "assistant", "content": GUIDE_MD},
        ]
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


def extract_cards(final_state: dict) -> dict:
    """그래프 final state에서 UI 표시용 cards dict로 변환."""
    cards = {}
    if final_state.get("results_rule"):
        cards["rule_based"] = final_state["results_rule"]
    if final_state.get("results_personalized"):
        cards["personalized"] = final_state["results_personalized"]
    if final_state.get("results_text"):
        cards["text_search"] = final_state["results_text"]
    if final_state.get("results_web"):
        cards["web_search"] = final_state["results_web"]
    return cards


def main():
    with st.sidebar:
        st.title("⚙️ 설정")
        page = st.radio("이동", ["🤖 챗봇", "📖 가이드"])
        st.divider()

        st.subheader("👥 플레이어 정보(빠방)")
        my_name = st.text_input("내 닉네임", placeholder="예: 코난", key="my_name_input")
        group_names = st.text_input(
            "같이 할 멤버 (옵션)", placeholder="예: 김전일, L", key="group_names_input"
        )

        nickname = my_name.strip()
        if group_names:
            nickname = (
                f"{nickname}, {group_names}".strip(", ") if nickname else group_names
            )

        if nickname:
            st.success(f"로그인: {nickname}")
        else:
            st.info("닉네임을 입력하면 맞춤 추천이 가능합니다.")

        st.divider()
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            # thread_id 새로 발급 → MemorySaver의 누적 상태도 단절
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    if page == "📖 가이드":
        st.markdown(GUIDE_MD)
        return

    st.title("🕵️ 방탈출 AI")
    st.caption("Hybrid Recommender System (LangGraph)")

    init_session_state()

    # 리소스 로드
    db = init_firebase()
    if not db:
        st.error("🔥 Firebase 연결 실패. 서비스 계정 키 또는 Secrets 설정을 확인하세요.")
        st.stop()
    if not GROQ_API_KEY:
        st.error("API Key가 설정되지 않았습니다.")
        st.stop()

    embed_model = load_embed_model()
    vec_rec = VectorRecommender(db, embed_model)
    rule_rec = RuleBasedRecommender(db)
    graph = compile_bot_graph(db, vec_rec, rule_rec, GROQ_API_KEY, TAVILY_API_KEY or "")

    # 채팅 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(msg)
            else:
                st.markdown(msg["content"])

    # 입력 처리
    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("🕵️ 테마를 추리 중입니다...", expanded=False) as status:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                try:
                    final_state = graph.invoke(
                        {"user_query": prompt, "user_context": nickname or None},
                        config=config,
                    )
                    status.update(label="추리 완료!", state="complete", expanded=False)
                except Exception as e:
                    logger.exception("그래프 실행 실패")
                    status.update(label=f"실패: {e}", state="error")
                    final_state = {"reply_text": f"❌ 실행 중 오류: {e}"}

            reply_text = final_state.get("reply_text", "응답 생성 실패")
            st.markdown(reply_text)
            cards = extract_cards(final_state)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply_text, "cards": cards}
        )
        st.rerun()


if __name__ == "__main__":
    main()
