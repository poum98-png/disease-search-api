import os
import json
from collections import defaultdict

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from supabase import create_client, Client

load_dotenv()

# =========================
# App / Env
# =========================
app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.getenv("SUPABASE_ANON_KEY", "").strip()
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) are required")

oa = OpenAI(api_key=OPENAI_API_KEY)
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CLASSIFY_MODEL = os.getenv("CLASSIFY_MODEL", "gpt-4.1-mini")
DEFAULT_TOPK = int(os.getenv("DEFAULT_TOPK", "5"))
DEFAULT_FETCH_K = int(os.getenv("FETCH_K", "30"))
CLASSIFY_MAX_UNITS = int(os.getenv("CLASSIFY_MAX_UNITS", "8"))


# =========================
# Heuristics for blocking / re-ask
# =========================
NON_SYMPTOM_KEYWORDS = [
    "가격", "비용", "얼마", "원", "만원", "견적",
    "보험", "실비", "청구", "적용", "서류",
    "예약", "진료시간", "시간", "영업", "휴무",
    "위치", "주소", "주차", "길찾기",
    "이벤트", "할인", "쿠폰",
    "후기", "리뷰",
    "원장", "의사", "스태프",
    "상담", "문의",
]

QUESTION_PATTERNS = [
    "뭐예요", "뭔가요", "무엇인가요", "왜 그런가요", "왜 이래요",
    "어떻게 하나요", "치료", "원인", "차이", "예방", "괜찮나요",
]

VAGUE_PHRASES = {
    "아파요", "아픈데", "통증", "시려요", "시림", "불편해요",
    "피나요", "붓기", "붓어요", "냄새", "입냄새", "고름",
    "치아", "잇몸", "이", "불편", "문제"
}

SYMPTOM_HINT_WORDS = [
    "치아", "이", "잇몸", "턱", "관자", "사랑니", "임플란트", "어금니", "앞니",
    "씹", "저작", "양치", "치실", "눌", "깨물",
    "찬", "뜨거", "단", "차가", "온도", "검게", "까맣", "변색", "구멍",
    "욱신", "찌릿", "쑤시", "시큰", "뻐근", "저리", "들뜨",
    "붓", "출혈", "피", "구취", "입냄새", "흔들", "깨짐", "갈라", "파절"
]


def looks_non_symptom(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    return any(k in q for k in NON_SYMPTOM_KEYWORDS)


def looks_question_like(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False
    if "?" in q:
        return True
    return any(k in q for k in QUESTION_PATTERNS)


def looks_too_vague(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return True

    if len(q) <= 3:
        return True

    if q in VAGUE_PHRASES:
        return True

    if len(q) <= 6 and not any(h in q for h in SYMPTOM_HINT_WORDS):
        return True

    return False


# =========================
# Utility
# =========================
def vec_to_pgvector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def embed_text(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        return []

    resp = oa.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding


def safe_json_from_model(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return json.loads(content)


# =========================
# LLM classify: split into semantic units
# =========================
CLASSIFY_SYSTEM_PROMPT = """
너는 치과 증상 검색을 위한 문장 분리기다.

목표:
- 사용자가 입력한 자유 텍스트를 "증상 의미 단위"로 분리한다.
- 각 단위의 text에는 반드시 원문 일부를 그대로 넣는다.
- 축약, 요약, 재작성, 교정, 번역을 하지 않는다.
- 입력에 없는 말을 보충하지 않는다.
- 한 단위가 너무 짧아 의미가 없어지지 않게 한다.
- 너무 잘게 쪼개지 않는다.
- 증상 검색이 아닌 일반 질문/상담/예약/가격 문의면 is_symptom_search=false 로 판단한다.
- 증상이라고 해도 너무 모호하면 needs_more_detail=true 로 판단한다.

판단 기준:
1) is_symptom_search
- true: 사용자가 자신의 증상/불편감/감각/겉모습/변화/유발상황을 말함
- false: 가격, 예약, 치료비, 진료시간, 위치, 일반 지식 질문, 원인/치료법 설명 요청 위주

2) needs_more_detail
- true 예시: "아파요", "시려요", "잇몸이요", "불편해요"
- false 예시: "오른쪽 아래 어금니가 찬물 마시면 찌릿하고 검게 보여요"

반드시 아래 JSON만 출력:
{
  "is_symptom_search": true,
  "needs_more_detail": false,
  "message": "",
  "units": [
    {
      "text": "..."
    }
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
            {"role": "user", "content": text},
        ],
    )

    obj = safe_json_from_model(resp.choices[0].message.content)

    units = obj.get("units", [])
    if not isinstance(units, list):
        units = []

    normalized_units = []
    seen = set()

    for u in units[:CLASSIFY_MAX_UNITS]:
        if not isinstance(u, dict):
            continue
        raw = str(u.get("text", "")).strip()
        if not raw:
            continue
        if raw in seen:
            continue
        seen.add(raw)
        normalized_units.append({"text": raw})

    return {
        "is_symptom_search": bool(obj.get("is_symptom_search", True)),
        "needs_more_detail": bool(obj.get("needs_more_detail", False)),
        "message": str(obj.get("message", "") or "").strip(),
        "units": normalized_units
    }


# =========================
# Supabase RPC search
# =========================
def rpc_match_symptom_statements(query_vec: list[float], match_count: int) -> list[dict]:
    if not query_vec:
        return []

    vec_literal = vec_to_pgvector_literal(query_vec)
    resp = sb.rpc(
        "match_symptom_statements",
        {
            "query_embedding": vec_literal,
            "match_count": match_count,
        }
    ).execute()

    return resp.data or []


def aggregate_results(unit_matches: list[dict], topk: int) -> list[dict]:
    """
    unit_matches item 예시:
    {
      "unit_text": "...",
      "matches": [
        {
          "row_id": "...",
          "disease_id": "...",
          "embed_txt": "...",
          "score": 0.81,
          "title": "...",
          "url": "..."
        }
      ]
    }
    """
    disease_bucket = defaultdict(lambda: {
        "title": "",
        "url": "",
        "all_scores": [],
        "matched_units": [],
        "matched_rows": []
    })

    for item in unit_matches:
        unit_text = item["unit_text"]
        rows = item["matches"]

        for r in rows:
            disease_id = r.get("disease_id")
            if not disease_id:
                continue

            score = float(r.get("score", 0.0) or 0.0)
            bucket = disease_bucket[disease_id]

            if not bucket["title"]:
                bucket["title"] = r.get("title", "") or ""
            if not bucket["url"]:
                bucket["url"] = r.get("url", "") or ""

            bucket["all_scores"].append(score)
            bucket["matched_units"].append({
                "unit_text": unit_text,
                "score": round(score, 6)
            })
            bucket["matched_rows"].append({
                "row_id": r.get("row_id", ""),
                "matched_text": r.get("embed_txt", ""),
                "score": round(score, 6),
                "unit_text": unit_text
            })

    results = []

    for disease_id, bucket in disease_bucket.items():
        row_scores = sorted(bucket["all_scores"], reverse=True)
        top_scores = row_scores[:2]
        disease_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        matched_rows_sorted = sorted(bucket["matched_rows"], key=lambda x: x["score"], reverse=True)
        matched_units_sorted = sorted(bucket["matched_units"], key=lambda x: x["score"], reverse=True)

        # 중복 제거
        seen_rows = set()
        dedup_rows = []
        for r in matched_rows_sorted:
            key = (r["row_id"], r["unit_text"])
            if key in seen_rows:
                continue
            seen_rows.add(key)
            dedup_rows.append(r)

        seen_units = set()
        dedup_units = []
        for u in matched_units_sorted:
            key = (u["unit_text"], u["score"])
            if key in seen_units:
                continue
            seen_units.add(key)
            dedup_units.append(u)

        best = dedup_rows[0] if dedup_rows else {"matched_text": "", "score": 0.0}

        results.append({
            "id": disease_id,
            "title": bucket["title"],
            "url": bucket["url"],
            "score_total": round(disease_score, 6),
            "best_match_text": best["matched_text"],
            "best_match_score": round(float(best["score"]), 6),
            "matched_units": dedup_units[:5],
            "top_matches": dedup_rows[:5],
        })

    results.sort(key=lambda x: x["score_total"], reverse=True)
    return results[:topk]


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

    if looks_non_symptom(text) or looks_question_like(text):
        return jsonify({
            "ok": True,
            "input": text,
            "is_symptom_search": False,
            "needs_more_detail": False,
            "message": "증상 검색용 입력이 아닙니다. 증상이나 불편감을 문장으로 적어주세요. 예: 오른쪽 아래 어금니가 찬물 마시면 찌릿해요",
            "units": []
        })

    if looks_too_vague(text):
        return jsonify({
            "ok": True,
            "input": text,
            "is_symptom_search": True,
            "needs_more_detail": True,
            "message": "증상을 조금 더 구체적으로 적어주세요. 예: 어디가 불편한지, 어떤 느낌인지, 언제 심해지는지 함께 적어주세요.",
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
    새 검색 로직:
    1) 사용자 자유 텍스트
    2) LLM으로 의미 단위 분리 (원문 그대로)
    3) 각 단위를 symptom_statements 대상으로 벡터검색
    4) 질환별 그룹핑 / 점수 집계
    """
    data = request.get_json(silent=True) or {}
    q = (data.get("text") or "").strip()

    if not q:
        return jsonify({"error": "text is required"}), 400

    if looks_non_symptom(q) or looks_question_like(q):
        return jsonify({
            "query": q,
            "blocked": True,
            "message": "증상 검색용 입력이 아닙니다. 가격, 예약, 위치, 일반 질문 대신 현재 느끼는 증상이나 변화를 적어주세요. 예: 치아가 검게 보이고 찬물 마시면 시려요",
            "results": [],
            "units": []
        }), 200

    if looks_too_vague(q):
        return jsonify({
            "query": q,
            "blocked": True,
            "message": "증상을 조금 더 자세히 적어주세요. 예: 어느 부위인지, 어떤 느낌인지, 언제 심해지는지 함께 적어주세요.",
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
            "message": cls["message"] or "증상 검색용 입력이 아닙니다. 현재 느끼는 증상을 문장으로 적어주세요.",
            "results": [],
            "units": []
        }), 200

    if cls["needs_more_detail"] or not cls["units"]:
        return jsonify({
            "query": q,
            "blocked": True,
            "message": cls["message"] or "증상을 조금 더 구체적으로 적어주세요. 예: 오른쪽 아래 어금니가 찬물 마시면 찌릿하고 검게 보여요",
            "results": [],
            "units": cls["units"]
        }), 200

    unit_matches = []

    for unit in cls["units"]:
        unit_text = unit["text"].strip()
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

    results = aggregate_results(unit_matches, topk=topk)

    return jsonify({
        "query": q,
        "blocked": False,
        "results": results,
        "units": cls["units"],
        "debug": {
            "fetch_k": fetch_k,
            "unit_count": len(cls["units"]),
        }
    })


@app.post("/admin/embed-symptom-statements")
def admin_embed_symptom_statements():
    """
    symptom_statements 테이블에서 embedding이 null인 행들을 임베딩으로 채움
    """
    token = request.args.get("token", "")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return jsonify({"error": "unauthorized"}), 401

    limit_n = int(request.args.get("limit", "300"))
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

