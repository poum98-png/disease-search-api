import os
import json
from typing import Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

# 모델 설정
CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", "gpt-4.1-mini")
DEFAULT_TOPK = int(os.getenv("DEFAULT_TOPK", "5"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 환경변수를 설정하세요.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")

app = Flask(__name__)
CORS(app)

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# 유틸
# =========================
def safe_json_loads(content: str) -> Any:
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


def normalize_relevance(v: str) -> str:
    v = (v or "").strip().lower()
    if v in ["high", "medium", "low"]:
        return v
    return "medium"


# =========================
# 질문 차단 로직
# - 증상 검색이 아닌 질문만 차단
# - 모호한 증상 입력은 통과
# =========================
NON_SYMPTOM_KEYWORDS = [
    "가격", "비용", "얼마", "원", "만원", "견적",
    "보험", "실비", "청구", "적용", "서류",
    "예약", "진료시간", "영업", "휴무",
    "위치", "주소", "주차", "길찾기",
    "이벤트", "할인", "쿠폰",
    "후기", "리뷰",
    "원장", "의사", "스태프",
    "상담", "문의", "전화", "연락",
    "치료비", "수술비"
]

QUESTION_ENDINGS = [
    "뭐예요", "뭔가요", "무엇인가요", "왜 그런가요", "왜 이래요",
    "어떻게 하나요", "어떻게 해요", "되나요", "가능한가요",
    "인가요", "나요", "일까요", "알 수 있나요"
]


def looks_non_symptom_question(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False

    if "?" in q and any(k in q for k in NON_SYMPTOM_KEYWORDS):
        return True

    if any(k in q for k in NON_SYMPTOM_KEYWORDS) and any(e in q for e in QUESTION_ENDINGS):
        return True

    if len(q) <= 20 and any(k in q for k in NON_SYMPTOM_KEYWORDS):
        return True

    return False


# =========================
# 의미 단위 분리 (선택)
# - 원문 그대로 유지
# - 질환 추론용 보조 정보
# =========================
CLASSIFY_SYSTEM_PROMPT = """
너는 치과 증상 검색용 문장 분리기다.

목표:
- 사용자가 입력한 자유 텍스트를 증상 의미 단위로 분리한다.
- 각 단위의 text에는 반드시 원문 일부를 그대로 넣는다.
- 축약, 요약, 교정, 재작성, 번역을 하지 않는다.
- 입력에 없는 말을 보충하지 않는다.
- 너무 잘게 쪼개지지 말고, 각 단위가 의미를 가지게 한다.
- 증상 검색이 아닌 일반 문의/가격/예약/위치/보험 질문이면 is_symptom_search=false 로 판단한다.

반드시 아래 JSON만 출력:
{
  "is_symptom_search": true,
  "units": [
    {"text": "..."}
  ]
}
"""


def classify_units(text: str) -> dict:
    resp = oa.chat.completions.create(
        model=CLASSIFY_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )

    obj = safe_json_loads(resp.choices[0].message.content)
    units = obj.get("units", [])
    if not isinstance(units, list):
        units = []

    normalized = []
    seen = set()

    max_units = int(os.getenv("CLASSIFY_MAX_UNITS", "8"))

    for u in units[:max_units]:
        if not isinstance(u, dict):
            continue
        t = str(u.get("text", "")).strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        normalized.append({"text": t})

    return {
        "is_symptom_search": bool(obj.get("is_symptom_search", True)),
        "units": normalized
    }


# =========================
# disease_clues 로드
# =========================
def load_disease_clues() -> list[dict]:
    resp = (
        sb.table("disease_clues")
        .select("disease_id, title, url, clues, clues_json")
        .eq("is_active", True)
        .execute()
    )

    rows = resp.data or []
    normalized = []

    for r in rows:
        clues_json = r.get("clues_json", [])
        if isinstance(clues_json, str):
            try:
                clues_json = json.loads(clues_json)
            except:
                clues_json = []

        if not isinstance(clues_json, list):
            clues_json = []

        normalized.append({
            "disease_id": r.get("disease_id", ""),
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "clues": r.get("clues", "") or "",
            "clues_json": [str(x).strip() for x in clues_json if str(x).strip()]
        })

    return normalized


# =========================
# LLM 질환 선택
# =========================
DISEASE_PICKER_SYSTEM_PROMPT = """
너는 치과 증상 큐레이션 검색용 질환 추천기다.

목표:
- 사용자의 증상/불편감 설명을 읽고,
- 제공된 질환 목록 안에서만,
- 관련 있을 수 있는 질환을 여러 개 고른다.

중요:
- 정확한 진단을 하려는 것이 아니다.
- 사용자의 증상으로 "나타날 수 있는 질환 후보"를 넓게 고른다.
- 완전히 무관한 질환은 넣지 않는다.
- 제공된 목록 밖의 질환명을 만들지 않는다.
- 입력이 모호해도 증상 기반이면 가능한 범위에서 관련 질환을 고른다.
- high / medium / low 중 하나로 relevance를 표시한다.
- 결과는 관련성 높은 순서대로 정렬한다.
- 최대 7개까지 반환한다.
- reason은 1문장으로 짧게 작성한다.

반드시 아래 JSON만 출력:
{
  "related_diseases": [
    {
      "disease_id": "...",
      "title": "...",
      "relevance": "high",
      "reason": "..."
    }
  ]
}
"""


def pick_related_diseases_with_llm(user_text: str, units: list[dict], disease_clues: list[dict]) -> dict:
    payload = {
        "user_input": user_text,
        "units": units,
        "disease_candidates": disease_clues
    }

    resp = oa.chat.completions.create(
        model=CLASSIFY_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DISEASE_PICKER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]
    )

    return safe_json_loads(resp.choices[0].message.content)


# =========================
# 관리자용 disease_clues 점검
# =========================
@app.post("/admin/check-disease-clues")
def admin_check_disease_clues():
    token = request.args.get("token", "")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify({"error": "unauthorized"}), 401

    try:
        rows = load_disease_clues()
        return jsonify({
            "ok": True,
            "count": len(rows),
            "sample": rows[:3]
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/classify")
def classify():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"ok": False, "error": "text is required"}), 400

    if looks_non_symptom_question(text):
        return jsonify({
            "ok": True,
            "input": text,
            "is_symptom_search": False,
            "message": "증상 검색용 입력이 아닙니다. 현재 느끼는 증상이나 변화를 적어주세요.",
            "units": []
        })

    try:
        result = classify_units(text)
        return jsonify({
            "ok": True,
            "input": text,
            **result
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"classify failed: {str(e)}"}), 500


@app.post("/smart-search")
def smart_search():
    """
    LLM 기반 관련 질환 열거:
    1) 증상 검색이 아닌 질문만 차단
    2) LLM으로 의미 단위 분리
    3) Supabase disease_clues 읽기
    4) LLM이 disease_clues 안에서 관련 질환 여러 개 선택
    """
    data = request.get_json(silent=True) or {}
    q = (data.get("text") or "").strip()

    if not q:
        return jsonify({"error": "text is required"}), 400

    if looks_non_symptom_question(q):
        return jsonify({
            "query": q,
            "blocked": True,
            "message": "증상 검색용 입력이 아닙니다. 가격, 예약, 위치, 보험 문의 대신 현재 느끼는 증상이나 변화를 적어주세요.",
            "results": []
        }), 200

    topk = int(data.get("k") or DEFAULT_TOPK)
    topk = max(1, min(20, topk))

    try:
        cls = classify_units(q)
    except Exception as e:
        return jsonify({"error": f"classify failed: {str(e)}"}), 500

    if not cls["is_symptom_search"]:
        return jsonify({
            "query": q,
            "blocked": True,
            "message": "증상 검색용 입력이 아닙니다. 현재 느끼는 증상을 문장으로 적어주세요.",
            "results": [],
            "units": []
        }), 200

    units = cls["units"]
    if not units:
        units = [{"text": q}]

    try:
        disease_clues = load_disease_clues()
    except Exception as e:
        return jsonify({
            "query": q,
            "blocked": True,
            "message": f"disease_clues load failed: {str(e)}",
            "results": [],
            "units": units
        }), 500

    if not disease_clues:
        return jsonify({
            "query": q,
            "blocked": False,
            "message": "질환 단서 데이터가 비어 있습니다.",
            "results": [],
            "units": units
        }), 200

    try:
        llm_result = pick_related_diseases_with_llm(q, units, disease_clues)
    except Exception as e:
        return jsonify({
            "query": q,
            "blocked": True,
            "message": f"llm disease search failed: {str(e)}",
            "results": [],
            "units": units
        }), 500

    raw_related = llm_result.get("related_diseases", [])
    if not isinstance(raw_related, list):
        raw_related = []

    # disease_clues metadata lookup
    meta_by_id = {x["disease_id"]: x for x in disease_clues}

    results = []
    for item in raw_related:
        if not isinstance(item, dict):
            continue

        disease_id = str(item.get("disease_id", "")).strip()
        if not disease_id:
            continue

        meta = meta_by_id.get(disease_id, {})

        title = str(item.get("title", "")).strip() or meta.get("title", "")
        relevance = normalize_relevance(str(item.get("relevance", "medium")))
        reason = str(item.get("reason", "")).strip()

        if relevance == "high":
            score = 0.9
        elif relevance == "medium":
            score = 0.6
        else:
            score = 0.3

        results.append({
            "id": disease_id,
            "title": title,
            "url": meta.get("url", ""),
            "relevance": relevance,
            "score_total": score,
            "reason": reason
        })

    # 중복 제거
    dedup = {}
    for r in results:
        did = r["id"]
        prev = dedup.get(did)
        if prev is None:
            dedup[did] = r
        else:
            # high > medium > low
            order = {"high": 3, "medium": 2, "low": 1}
            if order.get(r["relevance"], 0) > order.get(prev["relevance"], 0):
                dedup[did] = r

    final_results = list(dedup.values())
    final_results.sort(
        key=lambda x: (
            {"high": 3, "medium": 2, "low": 1}.get(x["relevance"], 0),
            x["title"]
        ),
        reverse=True
    )
    final_results = final_results[:topk]

    return jsonify({
        "query": q,
        "blocked": False,
        "results": final_results,
        "units": units,
        "debug": {
            "disease_clue_count": len(disease_clues),
            "unit_count": len(units)
        }
    })


# 선택: /llm-disease-search 별칭도 제공
@app.post("/llm-disease-search")
def llm_disease_search():
    return smart_search()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=True
    )
