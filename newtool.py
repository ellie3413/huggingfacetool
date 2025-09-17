import streamlit as st
import pandas as pd
from huggingface_hub import list_models
from datetime import datetime, timezone
import re
import time

# --- 1) 페이지 설정 ---
st.set_page_config(
    page_title="HF 모델 탐색 대시보드",
    page_icon="🤗",
    layout="wide",
)

# --- 2) 데이터 로딩 & 캐싱 ---
@st.cache_data(ttl=3600)
def search_models_on_hub(query: str, authors: tuple, sort: str, text_gen_only: bool):
    """
    사용자 필터 조건을 바탕으로 Hugging Face Hub API를 직접 호출합니다.
    """
    try:
        sort_map = {
            "다운로드순 (Downloads)": "downloads",
            "인기순 (Likes)": "likes",
            "최신순 (Created At)": "lastModified",
            "파라미터 크기순 (Parameter Size)": "downloads"  # API에서는 downloads로 가져오고 클라이언트에서 재정렬
        }
        api_sort = sort_map.get(sort, "downloads")
        api_filter = "text-generation" if text_gen_only else None

        # ✅ 1. 다중 회사 조회 오류 해결: 개별 조회 후 병합
        all_models_list = []
        target_authors = list(authors) if authors else [None]
        
        progress_bar = st.progress(0, text="모델 데이터를 가져오는 중...")
        for i, author in enumerate(target_authors):
            current_query = query if i == 0 or author is None else None

            models_chunk = list(list_models(
                search=current_query,
                author=author,
                filter=api_filter,
                sort=api_sort,
                limit=1000,
                full=True,
            ))
            all_models_list.extend(models_chunk)
            progress_bar.progress((i + 1) / len(target_authors), text=f"'{author or '전체'}' 모델 정보 로딩 완료...")
        
        progress_bar.empty()

        # 중복 제거
        seen_ids = set()
        unique_models = []
        for model in all_models_list:
            if model.modelId not in seen_ids:
                unique_models.append(model)
                seen_ids.add(model.modelId)
        
        models_list = unique_models

        processed = []
        for m in models_list:
            created_raw = getattr(m, "created_at", None) or getattr(m, "createdAt", None) or getattr(m, "lastModified", None)
            processed.append({
                "modelId": getattr(m, "modelId", None),
                "author": getattr(m, "author", None),
                "createdAt": created_raw,
                "downloads": getattr(m, "downloads", 0),
                "likes": getattr(m, "likes", 0),
                "pipeline_tag": getattr(m, "pipeline_tag", "N/A"),
                "tags": getattr(m, "tags", []) or [],
            })

        df = pd.DataFrame(processed)
        df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
        df["param_size"] = df.apply(lambda row: extract_param_size(row["tags"]) or extract_param_size_from_name(row["modelId"]), axis=1)
        df["param_size_numeric"] = df["param_size"].apply(param_size_to_numeric)
        return df

    except Exception as e:
        st.error(f"모델 데이터를 불러오는 데 실패했습니다: {e}")
        return pd.DataFrame()


# --- 3) tag 추출/판별 함수 ---
def extract_license(tags):
    for t in tags or []:
        if isinstance(t, str) and t.startswith("license:"): return t.split(":", 1)[-1]
    return None

def extract_task_from_tags(tags):
    KNOWN_TASKS = {"text-generation", "image-classification", "translation", "summarization", "question-answering"}
    for t in tags or []:
        if isinstance(t, str) and t in KNOWN_TASKS: return t
    return None

def param_size_to_numeric(param_size):
    if not param_size: return 0
    param_size = param_size.upper()
    moe_match = re.match(r"(\d+)X(\d+(?:\.\d+)?)B", param_size)
    if moe_match: return float(moe_match.group(1)) * float(moe_match.group(2))
    normal_match = re.match(r"(\d+(?:\.\d+)?)B", param_size)
    if normal_match: return float(normal_match.group(1))
    return 0

def extract_param_size_from_name(model_id):
    if not isinstance(model_id, str): return None
    match = re.search(r"[-_]?(\d+(?:\.\d+)?)[bB](?:[-_]|$)", model_id)
    if match: return f"{match.group(1)}B"
    match = re.search(r"[-_]?(\d+)x(\d+(?:\.\d+)?)[bB](?:[-_]|$)", model_id)
    if match: return f"{match.group(1)}x{match.group(2)}B"
    return None

def extract_param_size(tags):
    for t in tags or []:
        if isinstance(t, str):
            if re.match(r"^\d+(\.\d+)?[bB]$", t, re.IGNORECASE): return t.upper()
            if re.match(r"^\d+x\d+(\.\d+)?[bB]$", t, re.IGNORECASE): return t.upper()
    return None


# --- 4) UI 및 상태 관리 ---
st.title("🤗 Hugging Face 모델 탐색기 (실시간 검색)")

# --- 상태 초기화 ---
if "page" not in st.session_state: st.session_state.page = 1
if "query_input" not in st.session_state: st.session_state.query_input = ""
if "sort_input" not in st.session_state: st.session_state.sort_input = "다운로드순 (Downloads)"  # 기본값 다운로드순
if "search_params" not in st.session_state:
    st.session_state.search_params = {
        "query": "",
        "authors": [],
        "sort": "다운로드순 (Downloads)",  # 기본값 다운로드순
        "text_gen_only": True,
        "cutoff_date": pd.to_datetime("2024-01-01").date(),
        "param_range": (0.0, 1000.0)
    }

# --- 콜백 ---
def update_search():
    st.session_state.search_params["query"] = st.session_state.query_input
    st.session_state.search_params["sort"] = st.session_state.sort_input
    st.session_state.page = 1

# --- 사이드바 필터 ---
with st.sidebar:
    st.header("🔍 Filters")
    only_text_gen_widget = st.checkbox("text-generation 모델 보기", value=st.session_state.search_params["text_gen_only"], help="이 옵션을 켜면 HF tag에 'text-generation'이 포함된 모델만 표시됩니다. 일부 텍스트 생성 모델은 태그가 누락되어 목록에 보이지 않을 수 있습니다.")

    st.caption(
    "⚠️ 찾는 모델이 보이지 않으면 체크를 해제해 검색해 보시고, "
    "태그가 없더라도 텍스트 생성 모델일 수 있으니 HF의 모델 카드를 확인해 주십시오."
    )
    
    priority_authors = ["naver-hyperclovax", "google", "openai", "meta-llama", "mistralai", "microsoft", "Qwen", "deepseek-ai", "moonshotai", "zai.org", "baidu", "LGAI-EXAONE", "upstage", "kakaocorp", "skt", "K-intelligence"]
    authors_widget = st.multiselect("기업", options=priority_authors, default=st.session_state.search_params["authors"])
    

    cutoff_date_widget = st.date_input("기준 날짜 선택(모델 출시일)", value=st.session_state.search_params["cutoff_date"])

    st.subheader("📏 Parameter Size")
    param_range_widget = st.slider("Parameter Size 범위 (B)", 0.0, 1000.0, st.session_state.search_params["param_range"], 0.1)

    st.markdown("---")
    if st.button("🔄 필터 적용하기", use_container_width=True):
        st.session_state.search_params["authors"] = authors_widget
        st.session_state.search_params["text_gen_only"] = only_text_gen_widget
        st.session_state.search_params["cutoff_date"] = cutoff_date_widget
        st.session_state.search_params["param_range"] = param_range_widget
        st.session_state.page = 1
        st.rerun()

    # ✅ 색상 범례 (Legend)
    ORANGE = "rgba(255, 165, 0, 0.15)"   # 100만+
    YELLOW = "rgba(255, 255, 0, 0.15)"   # 50만~100만 미만
    PURPLE = "rgba(145, 97, 237, 0.14)"  # 국내 특정 조직 + 5만~50만 미만

    st.markdown("---")
    st.markdown("#### 색상 범례")
    st.markdown(f"""
    <style>
      .legend-item {{ display:flex; align-items:center; gap:8px; margin:6px 0; font-size:13px; }}
      .legend-dot  {{ width:12px; height:12px; border-radius:50%; border:1px solid rgba(0,0,0,.18); }}
      .legend-note {{ font-size:12px; opacity:.75; margin-top:4px; line-height:1.4; }}
    </style>
    <div class="legend-item">
      <span class="legend-dot" style="background:{ORANGE}"></span>
      <span>다운로드 100만 이상</span>
    </div>
    <div class="legend-item">
      <span class="legend-dot" style="background:{YELLOW}"></span>
      <span>다운로드 50만–100만</span>
    </div>
    <div class="legend-item">
      <span class="legend-dot" style="background:{PURPLE}"></span>
      <span>다운로드 5만 이상 (국내 모델)</span>
    </div>
    <div class="legend-note">
      국내 기업: naver-hyperclovax, kakaocorp, lgai-exaone, upstage, skt, K-intelligence
    </div>
    """, unsafe_allow_html=True)

# --- 메인 검색창/정렬 ---
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.text_input("Search by name", placeholder="🔎 모델 ID 또는 키워드로 검색 (예: meta-llama/Llama-2)", key="query_input", on_change=update_search)
with col2:
    st.selectbox("Sort by", ["다운로드순 (Downloads)", "인기순 (Likes)", "최신순 (Created At)", "파라미터 크기순 (Parameter Size)"], key="sort_input", on_change=update_search)

# --- 데이터 로딩 및 후처리 ---
search_args = st.session_state.search_params
base_df = search_models_on_hub(
    query=search_args["query"],
    authors=tuple(search_args["authors"]),
    sort=search_args["sort"],
    text_gen_only=search_args["text_gen_only"]
)

if base_df.empty:
    st.warning("데이터가 없거나 불러오는 데 실패했습니다. 필터를 조정하여 다시 검색해주세요.")
    st.stop()

# 클라이언트 사이드 후처리 필터
cutoff_ts = pd.Timestamp(search_args["cutoff_date"], tz="UTC")
final_df = base_df[(base_df["createdAt"].notna()) & (base_df["createdAt"] >= cutoff_ts)]
min_param, max_param = search_args["param_range"]
final_df = final_df[(final_df["param_size_numeric"] >= min_param) & (final_df["param_size_numeric"] <= max_param)]

# ✅ 정렬 로직 - 클라이언트에서 정확히 정렬
if search_args["sort"] == "다운로드순 (Downloads)":
    final_df = final_df.sort_values(by="downloads", ascending=False)
elif search_args["sort"] == "인기순 (Likes)":
    final_df = final_df.sort_values(by="likes", ascending=False)
elif search_args["sort"] == "최신순 (Created At)":
    final_df = final_df.sort_values(by="createdAt", ascending=False, na_position='last')
elif search_args["sort"] == "파라미터 크기순 (Parameter Size)":
    final_df = final_df.sort_values(by="param_size_numeric", ascending=True, na_position='last')  # 작은 모델부터

# --- 페이지네이션 및 출력 ---
st.metric(label="Filtered Models", value=f"{len(final_df):,}")
st.markdown("---")

MODELS_PER_PAGE = 15
total = len(final_df)
total_pages = max(1, (total + MODELS_PER_PAGE - 1) // MODELS_PER_PAGE)
if st.session_state.page > total_pages:
    st.session_state.page = total_pages
start_idx = (st.session_state.page - 1) * MODELS_PER_PAGE
end_idx = st.session_state.page * MODELS_PER_PAGE
page_df = final_df.iloc[start_idx:end_idx]

# --- 목록 출력 (특정 조직 5만~50만 → 연보라) ---
SPECIAL_AUTHORS = {
    "kakaocorp", "LGAI-EXAONE", "upstage", "skt", "K-intelligence", "naver-hyperclovax"
}

for _, row in page_df.iterrows():
    downloads = int(row["downloads"])
    likes = int(row["likes"])
    author = (row.get("author") or "").strip()

    # 1) 기본 배경/테두리
    bg_color = "rgba(0,0,0,0.00)"
    border_color = "rgba(0,0,0,0.06)"

    # 2) 특정 조직 + 다운로드 5만~50만 → 연보라
    if author in SPECIAL_AUTHORS and 50_000 <= downloads < 500_000:
        bg_color = "rgba(145, 97, 237, 0.14)"     # #9161ED with alpha
        border_color = "rgba(145, 97, 237, 0.35)"

    # 3) 일반 조건 (연보라 제외)
    elif downloads >= 1_000_000:                   # 100만 이상 → 주황
        bg_color = "rgba(255, 165, 0, 0.15)"
        border_color = "rgba(255, 165, 0, 0.35)"
    elif downloads >= 500_000:                     # 50만~100만 미만 → 노랑
        bg_color = "rgba(255, 255, 0, 0.15)"
        border_color = "rgba(255, 215, 0, 0.45)"

    created_str = row["createdAt"].strftime("%Y-%m-%d") if pd.notna(row["createdAt"]) else "N/A"
    license_str = extract_license(row["tags"])
    size_str = row["param_size"] or ""
    task_str = row.get("pipeline_tag", extract_task_from_tags(row["tags"]) or "N/A")
    region_label = "🇰🇷 국내 모델" if author in SPECIAL_AUTHORS else "🌏 해외 모델"

    meta_parts = []
    if size_str: meta_parts.append(f"📊 {size_str}")
    meta_parts.append(f"📅 Created {created_str}")
    if license_str: meta_parts.append(f"📄 License: {license_str}")
    if task_str and task_str != "N/A": meta_parts.append(f"🎯 {task_str}")
    if region_label: meta_parts.append(f"{region_label}")


    meta_html = " • ".join(meta_parts)

    card_html = f"""
    <div style="
        background:{bg_color};
        border:1px solid {border_color};
        border-radius:12px;
        padding:16px 18px;
        margin:12px 0;
    ">
      <div style="display:flex; gap:16px; align-items:flex-start;">
        <div style="flex:1 1 auto;">
          <h4 style="margin:0 0 6px 0; font-weight:500;">
            <a href="https://huggingface.co/{row['modelId']}" target="_blank" style="text-decoration:none;">
              {row['modelId']}
            </a>
          </h4>
          <div style="opacity:.8; font-size:13px; margin-top:2px;">
            {meta_html}
          </div>
        </div>
        <div style="min-width:140px; text-align:right; line-height:1.45;">
          <div>⬇️ {downloads:,}</div>
          <div>❤️ {likes:,}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)




# --- 페이지네이션 컨트롤 ---
if not final_df.empty:
    p_cols = st.columns([1, 2, 1])
    if p_cols[0].button("⬅️ Previous", disabled=(st.session_state.page <= 1)):
        st.session_state.page -= 1
        st.rerun()
    p_cols[1].write(
        f"<div style='text-align: center;'>Page <b>{st.session_state.page}</b> of <b>{total_pages}</b></div>",
        unsafe_allow_html=True
    )
    if p_cols[2].button("Next ➡️", disabled=(st.session_state.page >= total_pages)):
        st.session_state.page += 1
        st.rerun()
