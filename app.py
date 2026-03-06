import os
import json
from collections import defaultdict

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", "gpt-4.1-mini")
DEFAULT_TOPK = int(os.getenv("DEFAULT_TOPK", "5"))
DEFAULT_FETCH_K = int(os.getenv("FETCH_K", "30"))
CLASSIFY_MAX_UNITS = int(os.getenv("CLASSIFY_MAX_UNITS", "8"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 환경변수를 설정하세요.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")

app = Flask(__name__)
CORS(app)

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# 기본 유틸
# =========================
def embed_text(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        return []
    res = oa.embeddings.create(model=EMBED_MODEL, input=text)
    return res.data[0].embedding


def vec_to_pgvector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def safe_json_loads(content: str):
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


# =========================
# 질문 차단 로직
# - "증상 검색이 아닌 질문"만 차단
# - "모호한 입력"은 통과
# =========================
NON_SYMPTOM_QUESTION_KEYWORDS = [
    "가격", "비용", "얼마", "원", "만원", "견적",
    "보험", "실비", "청구", "적용", "서류",
    "예약", "진료시간", "시간", "영업", "휴무",
    "위치", "주소", "주차", "길찾기",
    "이벤트", "할인", "쿠폰",
    "후기", "리뷰",
    "원장", "의사", "스태프",
    "상담", "문의",
    "치료비", "치료가격"
]

QUESTION_ENDINGS = [
    "뭐예요", "뭔가요", "무엇인가요", "왜 그런가요", "왜 이래요",
    "어떻게 하나요", "어떻게 해요", "되나요", "가능한가요",
    "괜찮나요", "인가요", "나요"
]


def looks_non_symptom_question(q: str) -> bool:
    """
    증상 검색이 아닌 '질문성 문의'만 차단.
    모호한 증상 입력(예: 아파요, 시려요)은 차단하지 않음.
    """
    q = (q or "").strip()
    if not q:
        return False

    if "?" in q and any(k in q for k in NON_SYMPTOM_QUESTION_KEYWORDS):
        return True

    if any(k in q for k in NON_SYMPTOM_QUESTION_KEYWORDS):
        if any(e in q for e in QUESTION_ENDINGS):
            return True

    # 증상 단서 없이, 문의성 키워드만 있는 짧은 질문
    if len(q) <= 20 and any(k in q for k in NON_SYMPTOM_QUESTION_KEYWORDS):
        return True

    return False


# =========================
# 의미 단위 분리
# - 원문 그대로
# - 모호한 입력도 통과
# =========================
CLASSIFY_SYSTEM_PROMPT = """
너는 치과 증상 검색용 문장 분리기다.

목표:
- 사용자가 입력한 자유 텍스트를 증상 의미 단위로 분리한다.
- 각 단위의 text에는 반드시 원문 일부를 그대로 넣는다.
- 축약, 요약, 교정, 재작성, 번역을 하지 않는다.
- 입력에 없는 말을 보충하지 않는다.
- 너무 잘게 쪼개지지 말고, 각 단위가 의미를 가지게 한다.
- 모호한 입력이라도 가능한 범위에서 units를 만든다.
- 다만 증상 검색이 아닌 일반 문의/가격/예약/위치/보험 질문이면 is_symptom_search=false 로 판단한다.

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

    for u in units[:CLASSIFY_MAX_UNITS]:
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
# 검색 RPC
# Supabase 함수:
# match_symptom_statements(query_embedding vector(1536), match_count int)
# returns row_id, disease_id, embed_txt, score, title, url
# =========================
def rpc_match_symptom_statements(query_vec: list[float], match_count: int) -> list[dict]:
    if not query_vec:
        return []

    vec_literal = vec_to_pgvector_literal(query_vec)
    resp = sb.rpc(
        "match_symptom_statements",
        {
            "query_embedding": vec_literal,
            "match_count": match_count
        }
    ).execute()

    return resp.data or []


def aggregate_best_only(unit_matches: list[dict], topk: int) -> list[dict]:
    """
    질환별로 가장 유사한 문장 1개만 남김.
    그 1개 score로 Top-K 추천.
    """
    best_by_disease = {}

    for item in unit_matches:
        unit_text = item["unit_text"]
        matches = item["matches"]

        for r in matches:
            disease_id = r.get("disease_id")
            if not disease_id:
                continue

            score = float(r.get("score", 0.0) or 0.0)

            candidate = {
                "id": disease_id,
                "title": r.get("title", "") or "",
                "url": r.get("url", "") or "",
                "score_total": round(score, 6),
                "best_match_text": r.get("embed_txt", "") or "",
                "best_match_score": round(score, 6),
                "matched_unit": unit_text,
                "row_id": r.get("row_id", "") or ""
            }

            prev = best_by_disease.get(disease_id)
            if prev is None or score > prev["best_match_score"]:
                best_by_disease[disease_id] = candidate

    results = list(best_by_disease.values())
    results.sort(key=lambda x: x["best_match_score"], reverse=True)

    # 응답 정리
    final = []
    for r in results[:topk]:
        final.append({
            "id": r["id"],
            "title": r["title"],
            "url": r["url"],
            "score_total": r["score_total"],
            "best_match_text": r["best_match_text"],
            "best_match_score": r["best_match_score"],
            "matched_unit": r["matched_unit"],
            "row_id": r["row_id"]
        })
    return final


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

    # 질문만 차단, 모호한 입력은 통과
    if looks_non_symptom_question(text):
        return jsonify({
            "ok": True,
            "input": text,
            "is_symptom_search": False,
            "message": "증상 검색용 입력이 아닙니다. 현재 느끼는 증상이나 변화를 적어주세요. 예: 찬물 마시면 특정 치아가 찌릿해요",
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
    새 로직:
    1) 질문성 문의만 차단
    2) 모호한 입력은 그대로 통과
    3) LLM으로 의미 단위 분리
    4) 각 단위를 symptom_statements에서 벡터검색
    5) 질환별 '가장 높은 문장 1개'만 기준으로 Top-K 추천
    """
    data = request.get_json(silent=True) or {}
    q = (data.get("text") or "").strip()

    if not q:
        return jsonify({"error": "text is required"}), 400

    if looks_non_symptom_question(q):
        return jsonify({
            "query": q,
            "blocked": True,
            "message": "증상 검색용 입력이 아닙니다. 가격, 예약, 위치, 보험 문의 대신 현재 느끼는 증상을 적어주세요.",
            "results": [],
            "units": []
        }), 200

    topk = int(data.get("k") or DEFAULT_TOPK)
    topk = max(1, min(20, topk))

    fetch_k = int(data.get("fetch_k") or DEFAULT_FETCH_K)
    fetch_k = max(5, min(100, fetch_k))

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

    # 모호한 입력이라도 units가 하나도 없으면 전체 문장을 그대로 1개 unit로 사용
    if not units:
        units = [{"text": q}]

    unit_matches = []

    for unit in units:
        unit_text = (unit.get("text") or "").strip()
        if not unit_text:
            continue

        try:
            q_vec = embed_text(unit_text)
            rows = rpc_match_symptom_statements(q_vec, fetch_k)
        except Exception as e:
            return jsonify({"error": f"vector search failed: {str(e)}"}), 500

        unit_matches.append({
            "unit_text": unit_text,
            "matches": rows
        })

    results = aggregate_best_only(unit_matches, topk=topk)

    return jsonify({
        "query": q,
        "blocked": False,
        "results": results,
        "units": units,
        "debug": {
            "fetch_k": fetch_k,
            "unit_count": len(units),
            "unit_match_counts": [
                {
                    "unit_text": x["unit_text"],
                    "match_count": len(x["matches"])
                }
                for x in unit_matches
            ]
        }
    })


@app.post("/admin/embed-symptom-statements")
def admin_embed_symptom_statements():
    token = request.args.get("token", "")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify({"error": "unauthorized"}), 401

    limit_n = int(request.args.get("limit", "500"))
    limit_n = max(1, min(2000, limit_n))

    try:
        resp = (
            sb.table("symptom_statements")
            .select("row_id, embed_txt")
            .is_("embedding", "null")
            .limit(limit_n)
            .execute()
        )
        rows = resp.data or []

        updated = 0
        failed = []

        for r in rows:
            row_id = r.get("row_id")
            text = (r.get("embed_txt") or "").strip()

            if not text:
                failed.append({"row_id": row_id, "reason": "empty embed_txt"})
                continue

            try:
                vec = embed_text(text)
                sb.table("symptom_statements").update({
                    "embedding": vec_to_pgvector_literal(vec)
                }).eq("row_id", row_id).execute()
                updated += 1
            except Exception as e:
                failed.append({"row_id": row_id, "reason": str(e)})

        return jsonify({
            "total_candidates": len(rows),
            "updated": updated,
            "failed": failed
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=True
    )
