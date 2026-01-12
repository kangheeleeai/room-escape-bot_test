import streamlit as st
import time
import logging
from database import init_firebase
from models import load_embed_model

from recommenders import RuleBasedRecommender, VectorRecommender
from bot_engine import EscapeBotEngine
from config import GROQ_API_KEY, TAVILY_API_KEY

# --------------------------------------------------------------------------
# [로깅 설정] 앱 콘솔(터미널) 확인용
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="방탈출 AI (Hybrid)", page_icon="🕵️", layout="wide")

# CSS 스타일 (카드 디자인)
st.markdown("""
<style>
    .theme-card {
        background-color: #ffffff; /* 흰색 배경으로 변경 */
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); /* 흰색 배경 구분을 위한 옅은 그림자 */
        border: 1px solid #f0f0f0; /* 옅은 테두리 추가 */
    }
    .theme-title { 
        font-weight: bold; 
        font-size: 1.1em;
        color: #000000 !important; /* 제목 검은색 강제 지정 */
    }
    .theme-meta { font-size: 0.9em; color: #555; }
    .theme-desc { font-size: 0.9em; margin-top: 5px; color: #333; }
</style>
""", unsafe_allow_html=True)

def show_guide():
    """사용자 가이드 페이지"""
    st.markdown("""
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
    * 데이터 수집의 문제로 오래전에 빠방에 등록되었던 테마는 수집하지 못했습니다. 기록관리 기능으로 입력해두시면 영구적으로 기록됩니다.
    * 긴 요청사항은 AI가 이해하지 못합니다. 간결하게 작성하고 "다른거 추천해줘" 기능을 사용해주세요
    """)

def render_cards(card_list):
    """테마 카드 리스트를 렌더링하는 헬퍼 함수"""
    if not card_list:
        st.caption("결과가 없습니다.")
        return

    for item in card_list:
        # 설명이 없으면 빈 문자열 처리
        desc = item.get('desc', '')
        # [Fix] rating 변수 정의 (딕셔너리에서 안전하게 가져오기)
        rating = item.get('rating', 0.0)

        # 설명 길이 제한
        if len(desc) > 100:
            desc = desc[:100] + "..."
        
        # white-space: pre-wrap을 적용하여 줄바꿈을 유지하고 텍스트가 영역을 넘어갈 때 자동 줄바꿈되도록 함
        st.markdown(f"""
        <div class='theme-card'>
            <div class='theme-title'>{item['title']} <span style='font-size:0.8em; color:black'>({item['store']})</span></div>
            <div class='theme-meta'>⭐ 평점: {rating:.2f} | 📍 {item['location']}</div>
            <hr style="margin: 8px 0; opacity: 0.2;">
            <div class='theme-desc' style='white-space: pre-wrap; line-height: 1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    with st.sidebar:
        st.title("⚙️ 설정")
        
        page = st.radio("이동", ["🤖 챗봇", "📖 가이드"])
        st.divider()
        
        st.subheader("👥 플레이어 정보(빠방)")
        my_name = st.text_input("내 닉네임", placeholder="예: 코난", key="my_name_input")
        group_names = st.text_input("같이 할 멤버 (옵션)", placeholder="예: 김전일, L", key="group_names_input")
        
        nickname = my_name.strip()
        if group_names:
            nickname = f"{nickname}, {group_names}".strip(", ") if nickname else group_names

        if nickname:
            st.success(f"로그인: {nickname}")
        else:
            st.info("닉네임을 입력하면 맞춤 추천이 가능합니다.")
            
        st.divider()
        # debug_mode = st.toggle("🐛 디버그 모드", value=False, help="봇의 의도 분석 결과와 필터 정보를 보여줍니다.")
        
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.session_state.shown_theme_ids = set()
            st.session_state.last_filters = {}
            st.rerun()

    if page == "📖 가이드":
        show_guide()
        return

    # --- 메인 챗봇 화면 ---
    st.title("🕵️ 방탈출 AI")
    st.caption("Hybrid Recommender System")

    # Session State 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "어떤 방탈출 테마를 찾으시나요? 지역이나 장르를 말씀해주세요!"},
                                     {"role": "assistant", "content": """
                                                                        ## 🕵️ 방탈출 AI 사용 설명서
                                                                        
                                                                        ### 1️⃣ 기본 추천
                                                                        * "강남 공포 테마 추천해줘"
                                                                        * "홍대 활동성 많은거"
                                                                        
                                                                        ### 2️⃣ 닉네임 맞춤 추천
                                                                        * 왼쪽 사이드바에 닉네임을 입력하면 **내 플레이 기록**을 제외하고 추천합니다.
                                                                        * 친구들과 함께라면 쉼표(`,`)로 여러 명을 입력하세요.
                                                                        
                                                                        ### 3️⃣ 기록 관리
                                                                        * "**강남 링 했어**" -> 플레이 목록에 추가
                                                                        * "**홍대 삐릿뽀 안했어**" -> 기록 취소
                                                                        
                                                                        ### 0️⃣ 주의
                                                                        * 가능한 키워드: 공포(무서운, 안무서운 등), 연출, 인테리어, 스토리, 인원 관련, 문제방(어려운, 문제방, 안어려운 등), 활동성
                                                                        * 불가능 키워드: 판타지, SF, 코미디, 코믹, 서브 여부, 이머시브, 스릴러
                                                                        * 데이터 수집의 문제로 오래전에 빠방에 등록되었던 테마는 수집하지 못했습니다. 기록관리 기능으로 입력해두시면 영구적으로 기록됩니다.
                                                                        """}]
    if "shown_theme_ids" not in st.session_state:
        st.session_state.shown_theme_ids = set()
    if "last_filters" not in st.session_state:
        st.session_state.last_filters = {}

    # 리소스 로드
    db = init_firebase()
    embed_model = load_embed_model()

    if not db:
        st.error("🔥 Firebase 연결 실패. 서비스 계정 키 또는 Secrets 설정을 확인하세요.")
        st.stop()
    
    vec_rec = VectorRecommender(db, embed_model)
    rule_rec = RuleBasedRecommender(db) 
    bot_engine = EscapeBotEngine(vec_rec, rule_rec, GROQ_API_KEY, TAVILY_API_KEY)

    # 채팅 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            cards = msg.get("cards", {})
            debug_info = msg.get("debug_info", {})
            logs = msg.get("logs", [])

            # if logs:
            #     with st.expander("📜 처리 과정 로그 보기"):
            #         for l in logs:
            #             st.text(l)

            if cards:
                # 탭 구성
                tab1, tab2 = st.tabs(["🎯 맞춤 추천", "🔎 조건 추천"])
                
                with tab1:
                    # 맞춤 추천이 있으면 표시
                    if 'personalized' in cards:
                        render_cards(cards['personalized'])
                    else:
                        st.caption("맞춤 추천 결과가 없습니다. (로그인 필요)")

                with tab2:
                    # Rule-based 결과 표시
                    rule_list = cards.get('rule_based', [])
                    
                    # Fallback(유사검색) 결과를 여기에 합쳐서 보여줌
                    if not rule_list and 'text_search' in cards:
                        st.info("조건에 딱 맞는 테마가 없어 유사한 테마를 보여드립니다.")
                        rule_list = cards['text_search']
                    
                    if rule_list:
                        render_cards(rule_list)
                    else:
                        st.caption("검색 결과가 없습니다.")
            
            # if debug_mode and debug_info:
            #     with st.expander("🛠️ 디버그 정보"):
            #         st.json(debug_info)

    # 사용자 입력 처리
    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not GROQ_API_KEY:
                st.error("API Key가 설정되지 않았습니다.")
            else:
                process_logs = []
                with st.status("🕵️ 테마를 추리 중입니다...", expanded=True) as status:
                    
                    # 로그 콜백
                    # def ui_logger(msg):
                    #     st.write(f"🔹 {msg}")
                    #     process_logs.append(msg)
                    #     logger.info(msg)

                    session_ctx = {
                        'shown_ids': st.session_state.shown_theme_ids,
                        'last_filters': st.session_state.last_filters
                    }

                    # 봇 엔진 실행
                    reply_text, result_cards, used_filters, action, debug_data = bot_engine.generate_reply(
                        prompt, 
                        user_context=nickname,
                        session_context=session_ctx,
                        # on_log=ui_logger
                    )
                    
                    status.update(label="추리 완료!", state="complete", expanded=False)

                st.markdown(reply_text)
                
                # 중복 추천 방지 업데이트
                if result_cards:
                    if action == 'recommend': 
                        st.session_state.shown_theme_ids = set()
                    st.session_state.last_filters = used_filters
                    
                    for key in result_cards:
                        for c in result_cards[key]:
                            st.session_state.shown_theme_ids.add(c['id'])

        # 대화 저장
        st.session_state.messages.append({
            "role": "assistant", 
            "content": reply_text,
            "cards": result_cards,
            # "debug_info": debug_data if debug_mode else {},
            "logs": process_logs
        })
        st.rerun()

if __name__ == "__main__":
    main()
