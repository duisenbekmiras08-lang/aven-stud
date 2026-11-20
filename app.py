from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]


BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(exist_ok=True)
DB_PATH = INSTANCE_DIR / "aven_stud.sqlite3"
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY") or "local-demo-secret-change-before-deployment",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "0") == "1",
    MAX_CONTENT_LENGTH=256 * 1024,
    PERMANENT_SESSION_LIFETIME=60 * 60 * 8,
)

GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

raw_ollama_url = (
    os.getenv("OLLAMA_URL")
    or os.getenv("AI_BASE_URL")
    or "http://127.0.0.1:11434/v1"
).strip().rstrip("/")
if raw_ollama_url and not raw_ollama_url.endswith("/v1"):
    raw_ollama_url += "/v1"

OLLAMA_BASE_URL = raw_ollama_url
OLLAMA_API_KEY = (
    os.getenv("AI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or "ollama"
).strip() or "ollama"

if GROQ_API_KEY:
    AI_PROVIDER = "groq"
    AI_BASE_URL = GROQ_BASE_URL
    AI_API_KEY = GROQ_API_KEY
    AI_MODEL = (os.getenv("GROQ_MODEL") or os.getenv("AI_MODEL") or "openai/gpt-oss-20b").strip()
else:
    AI_PROVIDER = "ollama"
    AI_BASE_URL = OLLAMA_BASE_URL
    AI_API_KEY = OLLAMA_API_KEY
    AI_MODEL = (os.getenv("OLLAMA_MODEL") or os.getenv("AI_MODEL") or "gemma3:4b").strip()

MAJOR_DATA: dict[str, dict[str, Any]] = {
    "Бағдарламалық қамтамасыз ету": {
        "role": "it",
        "topics": [
            "Айнымалылар және типтер",
            "Шарттар мен циклдер",
            "Функциялар",
            "Массивтер және коллекциялар",
        ],
        "skills": [
            "Python / C#",
            "Алгоритмдер",
            "SQL және деректер қоры",
            "Git және командалық жұмыс",
            "REST API",
            "Тестілеу",
        ],
    },
    "Вайб кодинг": {
        "role": "vibecoding",
        "topics": [
            "Промпт-инжиниринг негіздері",
            "LLM және AI-ассистенттер",
            "Жылдам прототиптеу (MVP)",
            "Кодты тексеру және деплой",
        ],
        "skills": [
            "Cursor / AI IDE",
            "Промпт құрастыру",
            "API интеграциясы",
            "Python / JS негіздері",
            "Git және GitHub",
            "Өнімдік ойлау",
        ],
    },
    "Веб дизайн": {
        "role": "design",
        "topics": [
            "Композиция және торлар",
            "Типографика және түс",
            "UI/UX негіздері",
            "Адаптивті дизайн",
        ],
        "skills": [
            "Figma",
            "Прототиптеу",
            "Дизайн-жүйелер",
            "HTML / CSS негіздері",
            "UX зерттеу",
            "Визуалды коммуникация",
        ],
    },
}

DEFAULT_MAJOR = next(iter(MAJOR_DATA.keys()))

DEFAULT_STATE: dict[str, Any] = {
    "progress": {"tutor": 0, "language": 0, "simulator": 0, "wellbeing": 0, "career": 0},
    "metrics": {"foundation": 0, "english": 0, "practice": 0, "balance": 0, "career": 0},
    "tasks": [
        {"id": "task-1", "title": "Қайталау: негізгі тақырыпты бекіту", "done": False},
        {"id": "task-2", "title": "10 жаңа English терминін қайталау", "done": False},
        {"id": "task-3", "title": "Бір практикалық кейсті талдау", "done": False},
    ],
    "glossary": [],
    "bridges": [],
    "simulator": {"role": "it", "stage": 0, "trust": 65, "accuracy": 55, "stability": 70},
    "recall": {"seen": 0, "easy": 0},
    "tutorGapSolved": False,
    "aiCalibration": "Профиль деректері дайын. Жекелендіруді іске қосыңыз.",
}


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(_error: BaseException | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            pin_hash TEXT,
            profile_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS app_states (
            user_id INTEGER PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS ai_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            feature TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    conn.close()


init_db()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_response_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def get_user(user_id: int | None = None):
    uid = user_id or session.get("user_id")
    if not uid:
        return None
    return get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()


def normalize_major(major: str) -> str:
    return major if major in MAJOR_DATA else DEFAULT_MAJOR


def fresh_state_for_major(major: str) -> dict[str, Any]:
    state = json.loads(json.dumps(DEFAULT_STATE))
    role = MAJOR_DATA[normalize_major(major)]["role"]
    state["simulator"]["role"] = role
    return state


def load_state(user_id: int) -> dict[str, Any]:
    row = get_db().execute("SELECT state_json FROM app_states WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        return json.loads(json.dumps(DEFAULT_STATE))
    try:
        state = json.loads(row["state_json"])
        if not isinstance(state, dict):
            raise ValueError
    except (ValueError, TypeError):
        state = json.loads(json.dumps(DEFAULT_STATE))
    merged = json.loads(json.dumps(DEFAULT_STATE))
    merged.update(state)
    return merged


def save_state(user_id: int, state: dict[str, Any]) -> None:
    packed = json.dumps(state, ensure_ascii=False)
    get_db().execute(
        "INSERT INTO app_states(user_id, state_json, updated_at) VALUES(?, ?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at",
        (user_id, packed, now_iso()),
    )
    get_db().commit()


def public_user(row) -> dict[str, Any]:
    profile = json.loads(row["profile_json"])
    profile["major"] = normalize_major(profile.get("major", DEFAULT_MAJOR))
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "profile": profile,
    }


def auth_required(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return json_response_error("Сессия аяқталды. Қайта кіріңіз.", 401)
        if not get_user():
            session.clear()
            return json_response_error("Аккаунт табылмады. Қайта кіріңіз.", 401)
        return fn(*args, **kwargs)
    return wrapper


def csrf_required(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        expected = session.get("csrf_token")
        supplied = request.headers.get("X-CSRF-Token", "")
        if not expected or not supplied or not secrets.compare_digest(expected, supplied):
            return json_response_error("Қауіпсіздік токені жарамсыз. Бетті жаңартыңыз.", 403)
        return fn(*args, **kwargs)
    return wrapper


def issue_csrf() -> str:
    if not session.get("csrf_token"):
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def make_session(user_id: int) -> None:
    session.clear()
    session.permanent = True
    session["user_id"] = user_id
    session["csrf_token"] = secrets.token_urlsafe(32)


def validate_profile(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    name = str(payload.get("name", "")).strip()[:80]
    email = str(payload.get("email", "")).strip().lower()[:160]
    major = str(payload.get("major", DEFAULT_MAJOR))
    if major not in MAJOR_DATA:
        return None, "Оқу бағытын таңдаңыз."
    english = str(payload.get("english", "B1"))
    if english not in {"A1–A2", "B1", "B2", "C1–C2"}:
        return None, "Ағылшын деңгейін таңдаңыз."
    style = str(payload.get("style", "Ойындар және киберспорт"))
    if style not in {"Ойындар және киберспорт", "Спорт және тактика", "Музыка және ырғақ", "Академиялық база"}:
        return None, "Ойлау стилін таңдаңыз."
    goal = str(payload.get("goal", "Базадағы олқылықтарды жабу"))[:120]
    try:
        course = max(1, min(6, int(payload.get("course", 3))))
        work_hours = max(0, min(40, int(payload.get("workHours", 16))))
        days_to_exam = max(1, min(60, int(payload.get("daysToExam", 18))))
    except (TypeError, ValueError):
        return None, "Сандық мәндерді тексеріңіз."
    if len(name) < 2:
        return None, "Атыңызды енгізіңіз."
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        return None, "Email форматын тексеріңіз."
    return {
        "name": name,
        "email": email,
        "major": major,
        "course": course,
        "english": english,
        "workHours": work_hours,
        "daysToExam": days_to_exam,
        "style": style,
        "goal": goal,
        "pinEnabled": bool(payload.get("pinEnabled")),
    }, None


def ai_limit_reached(user_id: int, limit: int = 20) -> bool:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    row = get_db().execute(
        "SELECT COUNT(*) AS total FROM ai_logs WHERE user_id = ? AND created_at >= ?",
        (user_id, cutoff),
    ).fetchone()
    return int(row["total"]) >= limit


def log_ai_call(user_id: int, feature: str) -> None:
    get_db().execute(
        "INSERT INTO ai_logs(user_id,feature,created_at) VALUES(?,?,?)",
        (user_id, feature, now_iso()),
    )
    get_db().commit()


def local_calibration(profile: dict[str, Any]) -> str:
    english_mode = (
        "қарапайым аударма және терминдердің түсіндірмесі"
        if profile["english"] in {"A1–A2", "B1"}
        else "түпнұсқа академиялық терминология"
    )
    daily = max(1.0, round(5.0 - profile["workHours"] * 0.055 - max(0, 14 - profile["daysToExam"]) * 0.08, 1))
    return (
        f"{profile['major']} · {profile['course']}-курс профилі таңдалды. "
        f"Ағылшын {profile['english']} деңгейіне сай {english_mode} қосылды. "
        f"Түсіндіру стилі: {profile['style']}. Ұсынылатын терең оқу уақыты: шамамен {daily} сағ/күн."
    )


def ai_configured() -> bool:
    return OpenAI is not None and bool(AI_BASE_URL and AI_API_KEY)


def openai_text(instructions: str, prompt: str, max_output_tokens: int = 700) -> str:
    if not ai_configured():
        raise RuntimeError("AI_NOT_CONFIGURED")
    client = OpenAI(
        base_url=AI_BASE_URL,
        api_key=AI_API_KEY,
        timeout=45.0,
    )
    response = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_output_tokens,
        temperature=0.7,
    )
    result = (response.choices[0].message.content or "").strip()
    if not result:
        raise RuntimeError("EMPTY_AI_RESPONSE")
    return result


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.get("/")
def index():
    return render_template("index.html", major_data=MAJOR_DATA)


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "database": "SQLite",
        "aiConfigured": ai_configured(),
        "provider": AI_PROVIDER,
        "model": AI_MODEL if ai_configured() else None,
        "baseUrl": AI_BASE_URL,
        "majors": MAJOR_DATA,
    })


@app.post("/api/auth/register")
def register():
    payload = request.get_json(silent=True) or {}
    profile, error = validate_profile(payload)
    if error:
        return json_response_error(error)
    password = str(payload.get("password", ""))
    if (
        len(password) < 8
        or not re.search(r"[A-ZА-ЯӘІҢҒҮҰҚӨҺ]", password)
        or not re.search(r"\d", password)
        or not re.search(r"[^A-Za-zА-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ0-9]", password)
    ):
        return json_response_error("Құпиясөз кемінде 8 таңба, бас әріп, сан және арнайы таңба қамтуы керек.")
    pin_enabled = bool(payload.get("pinEnabled"))
    pin = str(payload.get("pin", ""))
    if pin_enabled and not re.fullmatch(r"\d{4}", pin):
        return json_response_error("2FA PIN дәл 4 саннан тұруы керек.")
    db = get_db()
    try:
        cur = db.execute(
            "INSERT INTO users(name,email,password_hash,pin_hash,profile_json,created_at) VALUES(?,?,?,?,?,?)",
            (
                profile["name"],
                profile["email"],
                generate_password_hash(password, method="scrypt"),
                generate_password_hash(pin, method="scrypt") if pin_enabled else None,
                json.dumps(profile, ensure_ascii=False),
                now_iso(),
            ),
        )
        user_id = cur.lastrowid
        save_state(user_id, fresh_state_for_major(profile["major"]))
        make_session(user_id)
    except sqlite3.IntegrityError:
        return json_response_error("Бұл email бұрын тіркелген. Кіру бөлімін пайдаланыңыз.", 409)
    row = get_user(user_id)
    return jsonify({
        "user": public_user(row),
        "state": load_state(user_id),
        "csrf": issue_csrf(),
        "message": "Қорғалған профиль құрылды.",
    }), 201


@app.post("/api/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))
    row = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        return json_response_error("Email немесе құпиясөз қате.", 401)
    make_session(row["id"])
    return jsonify({
        "user": public_user(row),
        "state": load_state(row["id"]),
        "csrf": issue_csrf(),
        "message": "Сәтті кірдіңіз.",
    })


@app.post("/api/auth/demo")
def demo_login():
    db = get_db()
    email = "demo@aven.local"
    row = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not row:
        profile = {
            "name": "Алихан",
            "email": email,
            "major": DEFAULT_MAJOR,
            "course": 3,
            "english": "B1",
            "workHours": 16,
            "daysToExam": 18,
            "style": "Ойындар және киберспорт",
            "goal": "Практикаға және тағылымдамаға дайындалу",
            "pinEnabled": True,
        }
        cur = db.execute(
            "INSERT INTO users(name,email,password_hash,pin_hash,profile_json,created_at) VALUES(?,?,?,?,?,?)",
            (
                profile["name"],
                email,
                generate_password_hash(secrets.token_urlsafe(24), method="scrypt"),
                generate_password_hash("1234", method="scrypt"),
                json.dumps(profile, ensure_ascii=False),
                now_iso(),
            ),
        )
        user_id = cur.lastrowid
        save_state(user_id, fresh_state_for_major(DEFAULT_MAJOR))
    else:
        user_id = row["id"]
    make_session(user_id)
    row = get_user(user_id)
    return jsonify({
        "user": public_user(row),
        "state": load_state(user_id),
        "csrf": issue_csrf(),
        "message": "Демо-профиль ашылды. PIN: 1234",
    })


@app.post("/api/auth/logout")
def logout():
    session.clear()
    return jsonify({"message": "Сіз аккаунттан шықтыңыз."})


@app.post("/api/auth/unlock")
@auth_required
@csrf_required
def unlock():
    payload = request.get_json(silent=True) or {}
    pin = str(payload.get("pin", ""))
    password = str(payload.get("password", ""))
    row = get_user()
    if not row:
        return json_response_error("Аккаунт табылмады.", 401)
    if row["pin_hash"]:
        valid = check_password_hash(row["pin_hash"], pin)
    else:
        valid = check_password_hash(row["password_hash"], password)
    if not valid:
        return json_response_error("PIN немесе құпиясөз қате. Қайта көріңіз.", 401)
    return jsonify({"message": "Экран ашылды."})


@app.get("/api/me")
@auth_required
def me():
    row = get_user()
    return jsonify({"user": public_user(row), "state": load_state(row["id"]), "csrf": issue_csrf()})


@app.put("/api/profile")
@auth_required
@csrf_required
def update_profile():
    payload = request.get_json(silent=True) or {}
    old = json.loads(get_user()["profile_json"])
    payload = {**old, **payload, "email": old["email"], "name": payload.get("name", old["name"])}
    profile, error = validate_profile(payload)
    if error:
        return json_response_error(error)
    pin = str(payload.get("pin", ""))
    pin_enabled = bool(payload.get("pinEnabled", bool(get_user()["pin_hash"])))
    if pin and not re.fullmatch(r"\d{4}", pin):
        return json_response_error("PIN дәл 4 сан болуы керек.")
    pin_hash = generate_password_hash(pin, method="scrypt") if pin else get_user()["pin_hash"]
    if not pin_enabled:
        pin_hash = None
    get_db().execute(
        "UPDATE users SET name=?, profile_json=?, pin_hash=? WHERE id=?",
        (profile["name"], json.dumps(profile, ensure_ascii=False), pin_hash, session["user_id"]),
    )
    get_db().commit()

    state = load_state(session["user_id"])
    new_role = MAJOR_DATA[profile["major"]]["role"]
    if state.get("simulator", {}).get("role") != new_role:
        state["simulator"] = {"role": new_role, "stage": 0, "trust": 65, "accuracy": 55, "stability": 70}
        save_state(session["user_id"], state)

    return jsonify({
        "user": public_user(get_user()),
        "state": state,
        "message": "ИИ-профиль жаңартылды.",
    })


@app.put("/api/state")
@auth_required
@csrf_required
def update_state():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return json_response_error("Күй деректері қате.")
    if len(json.dumps(payload, ensure_ascii=False)) > 180_000:
        return json_response_error("Сақталатын дерек тым үлкен.", 413)
    current = load_state(session["user_id"])
    current.update(payload)
    save_state(session["user_id"], current)
    return jsonify({"state": current, "saved": True})


@app.post("/api/ai/calibrate")
@auth_required
@csrf_required
def ai_calibrate():
    row = get_user()
    profile = json.loads(row["profile_json"])
    profile["major"] = normalize_major(profile.get("major", DEFAULT_MAJOR))
    local = local_calibration(profile)
    if not ai_configured():
        state = load_state(row["id"])
        state["aiCalibration"] = local + " (Жергілікті калибрлеу: ИИ бапталмаған.)"
        save_state(row["id"], state)
        return jsonify({"text": state["aiCalibration"], "mode": "local"})
    if ai_limit_reached(row["id"]):
        return json_response_error("Сағаттық ИИ сұрау лимиті аяқталды. Кейінірек қайталап көріңіз.", 429)
    prompt = (
        "Create a concise personalization plan for a college/university student. "
        "Use the profile below. Output in Kazakh, 3-5 short sentences. Never infer medical or sensitive traits.\n\n"
        + json.dumps(profile, ensure_ascii=False)
    )
    instructions = (
        "You are AvenAI Calibrator, a study-planning assistant for Polytech · AVEN Stud. "
        "Adapt language complexity, analogies, practice scenario and workload suggestions to the student's profile. "
        "Be realistic and supportive. Do not claim to have measured cognitive ability or diagnose burnout. "
        "Treat workload calculations as estimates."
    )
    try:
        text = openai_text(instructions, prompt, 420)
        state = load_state(row["id"])
        state["aiCalibration"] = text
        save_state(row["id"], state)
        log_ai_call(row["id"], "calibration")
        return jsonify({"text": text, "mode": "openai", "provider": AI_PROVIDER})
    except Exception as exc:
        app.logger.warning("AI calibration failed: %s", exc)
        return jsonify({"text": local + f" ({AI_PROVIDER} уақытша қолжетімсіз.)", "mode": "local"})


@app.post("/api/ai/chat")
@auth_required
@csrf_required
def ai_chat():
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt", "")).strip()
    feature = str(payload.get("feature", "tutor"))[:40]
    if len(prompt) < 2:
        return json_response_error("Сұрағыңызды жазыңыз.")
    if len(prompt) > 3000:
        return json_response_error("Сұрақ 3000 таңбадан аспауы керек.")
    row = get_user()
    profile = json.loads(row["profile_json"])
    profile["major"] = normalize_major(profile.get("major", DEFAULT_MAJOR))
    if not ai_configured():
        return json_response_error("ИИ қосылмаған. GROQ_API_KEY немесе OLLAMA_URL параметрін тексеріңіз.", 503)
    if ai_limit_reached(row["id"]):
        return json_response_error("Сағаттық ИИ сұрау лимиті аяқталды. Кейінірек қайталап көріңіз.", 429)
    feature_context = {
        "tutor": "Personal tutor: diagnose misunderstanding, explain step by step, ask a check question.",
        "language": "Academic English coach: explain vocabulary, translate or simplify academic text at the student's CEFR level.",
        "simulator": "Practice mentor: evaluate a student's decision in a simulated professional case; teach rather than provide a shortcut.",
        "wellbeing": "Study planner: suggest manageable micro-steps and a balanced plan. Never diagnose health conditions.",
        "career": "Career coach: map skills to entry-level study milestones. Do not invent job-market statistics or citations.",
    }.get(feature, "Study mentor")
    instructions = (
        "You are AvenAI, an educational mentor for a Polytech college student. Respond in Kazakh unless the user explicitly asks otherwise. "
        "Your goal is to strengthen the student's own thinking, not replace it. Do not write complete ready-to-submit homework, "
        "exam answers, essays, or projects. Instead give hints, explain concepts, ask guiding questions, and help the learner "
        "work through one step at a time. If they ask to copy a full solution, kindly refuse and offer a first hint. "
        "Adapt to the profile: " + json.dumps(profile, ensure_ascii=False) + ". "
        + feature_context + " Do not fabricate citations, URLs, publications, or claims of source verification. "
        "When uncertain, say so and suggest what source or fact should be checked. Keep the answer structured and concise."
    )
    try:
        text = openai_text(instructions, prompt, 750)
        log_ai_call(row["id"], feature)
        return jsonify({
            "text": text,
            "mode": "openai",
            "provider": AI_PROVIDER,
            "sourceNote": f"ИИ-жауап ({AI_PROVIDER} · {AI_MODEL}) · сыртқы дереккөздер автоматты түрде расталмайды",
        })
    except Exception as exc:
        app.logger.warning("AI request failed: %s", exc)
        return json_response_error(f"ИИ қатесі ({AI_PROVIDER}): {exc}", 502)


@app.errorhandler(413)
def too_large(_error):
    return json_response_error("Сұрау көлемі тым үлкен.", 413)


if __name__ == "__main__":
    print("Polytech · AVEN Stud: http://127.0.0.1:5000")
    print(f"AI Provider: {AI_PROVIDER} | Endpoint: {AI_BASE_URL} | Model: {AI_MODEL}")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "0") == "1")
