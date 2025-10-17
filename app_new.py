# app_new.py
"""
Lightweight, testable version of the AI Web App Builder.

Features:
- POST /ready : accept task requests, verify secret, schedule background job
- POST /evaluation-receiver : accept evaluator notifications (for testing)
- background job: generate simple files (index.html, README.md, LICENSE)
- save attachments (data: URIs or http URLs)
- DRY_RUN mode: skip LLM + GitHub push (safe for local testing)
- optional GitHub push if GITHUB_TOKEN and GITHUB_USERNAME provided and DRY_RUN is false

Run:
  uvicorn app_new:app --reload
"""

import os
import re
import json
import base64
import shutil
import asyncio
import logging
import sys
from typing import List, Optional
from datetime import datetime

import httpx
import git
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# -----------------------
# Settings
# -----------------------
class Settings(BaseSettings):
    STUDENT_SECRET: str = Field("test-secret", env="STUDENT_SECRET")
    GITHUB_TOKEN: str = Field("", env="GITHUB_TOKEN")
    GITHUB_USERNAME: str = Field("", env="GITHUB_USERNAME")
    DRY_RUN: bool = Field(True, env="DRY_RUN")
    LOG_FILE_PATH: str = Field("/tmp/app.log", env="LOG_FILE_PATH")
    GITHUB_API_BASE: str = Field("https://api.github.com", env="GITHUB_API_BASE")
    AIPIPE_API_KEY: str = Field(..., env="AIPIPE_API_KEY")  # required
    AIPIPE_BASE: str = Field("https://aipipe.org/openrouter/v1", env="AIPIPE_BASE")
    AIPIPE_MODEL: str = Field("openai/gpt-4.1-nano", env="AIPIPE_MODEL")
    AIPIPE_TIMEOUT_SECONDS: int = Field(90, env="AIPIPE_TIMEOUT_SECONDS")
    GITHUB_PAGES_BASE: Optional[str] = Field(None, env="GITHUB_PAGES_BASE")
    MAX_CONCURRENT_TASKS: int = Field(2, env="MAX_CONCURRENT_TASKS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "forbid"  # avoid "extra inputs are not permitted" errors


settings = Settings()
if not settings.GITHUB_PAGES_BASE and settings.GITHUB_USERNAME:
    settings.GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"

# -----------------------
# Logging
# -----------------------
os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
logger = logging.getLogger("app_new")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
fh = logging.FileHandler(settings.LOG_FILE_PATH, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers = []
logger.addHandler(ch)
logger.addHandler(fh)
logger.propagate = False

def flush_logs():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for h in logger.handlers:
            try: h.flush()
            except Exception: pass
    except Exception:
        pass

# -----------------------
# Models
# -----------------------
class Attachment(BaseModel):
    name: str
    url: str

class TaskRequest(BaseModel):
    task: str
    email: str
    round: int
    brief: str
    evaluation_url: str
    nonce: str
    secret: str
    attachments: List[Attachment] = []

# -----------------------
# App & globals
# -----------------------
app = FastAPI(title="AI Web App Builder (Simple)")
task_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
background_tasks_list: List[asyncio.Task] = []
last_received_task: Optional[dict] = None

# -----------------------
# Helpers
# -----------------------
def verify_secret(s: str) -> bool:
    return s == settings.STUDENT_SECRET

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

async def save_attachments_locally(task_dir: str, attachments: List[Attachment]) -> List[str]:
    saved = []
    async with httpx.AsyncClient(timeout=30) as client:
        for att in attachments:
            try:
                if att.url.startswith("data:"):
                    m = re.search(r"base64,(.*)", att.url, re.IGNORECASE)
                    if not m:
                        continue
                    data = base64.b64decode(m.group(1))
                else:
                    resp = await client.get(att.url)
                    resp.raise_for_status()
                    data = resp.content
                p = os.path.join(task_dir, att.name)
                await asyncio.to_thread(lambda p, d: open(p, "wb").write(d), p, data)
                saved.append(att.name)
                logger.info(f"Saved attachment {att.name}")
            except Exception as e:
                logger.warning(f"Failed to save attachment {att.name}: {e}")
    flush_logs()
    return saved

def sanitize_repo_name(task_id: str) -> str:
    name = task_id.replace(" ", "-").lower()
    name = re.sub(r"[^a-z0-9._-]", "-", name)[:100]
    return name

def write_file(path: str, contents: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(contents)

async def notify_evaluator(evaluation_url: str, payload: dict) -> bool:
    if not evaluation_url or "example.com" in evaluation_url:
        logger.info("Skipping notify: invalid evaluation_url")
        return False
    max_retries = 3
    delay = 1
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(evaluation_url, json=payload)
                r.raise_for_status()
                logger.info("Notification sent to evaluator")
                return True
        except Exception as e:
            logger.warning(f"Notify attempt {attempt+1} failed: {e}")
            await asyncio.sleep(delay)
            delay *= 2
    logger.error("Notify failed after retries")
    return False

# -----------------------
# Core: generate files (simple local generator)
# -----------------------
def local_generate_files(task_id: str, brief: str, attachments: List[Attachment]) -> dict:
    year = datetime.utcnow().year
    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{task_id} - Demo</title>
  <style>body{{font-family:system-ui,Segoe UI,Roboto,Arial;max-width:900px;margin:2rem auto;padding:1rem}} .kbd{{font-family:monospace;background:#f1f1f1;padding:.1rem .3rem;border-radius:4px}}</style>
</head>
<body>
  <h1>{task_id}</h1>
  <p><strong>Brief:</strong> {brief}</p>
  <section id="app">
    <p>This is a generated demo page. If a <code>?url=</code> parameter is present, the app will try to show the image.</p>
    <div>
      <img id="preview" alt="preview" style="max-width:100%;border:1px solid #ddd;padding:6px;display:none"/>
      <p id="msg"></p>
    </div>
    <div>
      <label>Manual input: <input id="captchatxt" /></label>
      <button id="submit">Submit</button>
      <div id="result"></div>
    </div>
  </section>
  <script>
    const params = new URLSearchParams(location.search);
    const url = params.get('url');
    const preview = document.getElementById('preview');
    const msg = document.getElementById('msg');
    if (url) {{
      preview.src = url;
      preview.style.display = 'block';
      preview.onerror = ()=> msg.textContent = 'Failed to load remote URL, fallback to sample if available.';
    }} else {{
      msg.textContent = 'No ?url= parameter provided. Using sample if present.';
    }}
    document.getElementById('submit').addEventListener('click', ()=> {{
      const val = document.getElementById('captchatxt').value.trim();
      if (!val) return alert('Enter something');
      // simple fake check for demo
      const ok = val.toLowerCase().includes('test') || val.length>3;
      document.getElementById('result').textContent = ok ? 'SUCCESS' : 'FAIL';
    }});
  </script>
</body>
</html>"""
    readme = f"# {task_id}\n\nGenerated demo for brief:\n\n{brief}\n\nAttachments included: {', '.join([a.name for a in attachments]) or 'none'}\n"
    license_text = f"MIT License\n\nCopyright (c) {year}\n\nPermission is hereby granted..."
    return {"index.html": index_html, "README.md": readme, "LICENSE": license_text}










# -----------------------
# LLM / Aipipe integration (function-call tool pattern)
# -----------------------
import json
from openai import OpenAI as OpenAIClient  # pip package 'openai'

# Tool schema for the model to return files via a function call
GENERATED_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_files",
        "description": "Return a JSON object with keys: index.html, README.md, LICENSE containing full file contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "index.html": {"type": "string"},
                "README.md": {"type": "string"},
                "LICENSE": {"type": "string"}
            },
            "required": ["index.html", "README.md", "LICENSE"],
            "additionalProperties": False
        }
    }
}

async def _build_attachment_parts_for_llm(attachments: List[Attachment]) -> List[dict]:
    """
    Convert attachments to model-friendly parts:
     - Images -> inlineData: {mimeType, data (base64)}
     - Small text files -> text blocks
     - Large files ignored (but listed via metadata)
    """
    parts = []
    metadata_lines = []
    async with httpx.AsyncClient(timeout=30) as client:
        for att in attachments:
            metadata_lines.append(f"{att.name} - url: {att.url}")
            try:
                if att.url.startswith("data:"):
                    # keep as-is: data:<mime>;base64,<b64>
                    m = re.search(r"data:(?P<mime>[^;]+);base64,(?P<data>.*)", att.url, re.IGNORECASE)
                    if m:
                        mime = m.group("mime")
                        b64 = m.group("data")
                        if mime.startswith("image/"):
                            parts.append({"inlineData": {"mimeType": mime, "data": b64, "name": att.name}})
                        else:
                            decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
                            if len(decoded) > 20000:
                                decoded = decoded[:20000] + "\n\n...TRUNCATED..."
                            parts.append({"type": "text", "text": f"ATTACHMENT ({att.name} - {mime}):\n{decoded}"})
                    else:
                        # unparseable data: URI -> include metadata only
                        parts.append({"type": "text", "text": f"ATTACHMENT (unparsed data URI): {att.name}"})
                else:
                    # remote URL -> try fetch small text or image
                    try:
                        r = await client.get(att.url, timeout=20)
                        r.raise_for_status()
                        mime = r.headers.get("Content-Type", "application/octet-stream")
                        content = r.content
                        if mime and mime.startswith("image/"):
                            b64 = base64.b64encode(content).decode("utf-8")
                            parts.append({"inlineData": {"mimeType": mime, "data": b64, "name": att.name}})
                        else:
                            # treat as text when possible
                            try:
                                decoded = content.decode("utf-8", errors="ignore")
                                if len(decoded) > 20000:
                                    decoded = decoded[:20000] + "\n\n...TRUNCATED..."
                                parts.append({"type": "text", "text": f"ATTACHMENT ({att.name} - {mime}):\n{decoded}"})
                            except Exception:
                                # binary but non-image -> metadata only
                                parts.append({"type": "text", "text": f"ATTACHMENT (binary): {att.name} - {mime}"})
                    except Exception as e:
                        parts.append({"type": "text", "text": f"ATTACHMENT (fetch failed): {att.name} - {str(e)}"})
            except Exception as e:
                parts.append({"type": "text", "text": f"ATTACHMENT (error processing): {att.name} - {str(e)}"})
    if metadata_lines:
        parts.append({"type": "text", "text": "ATTACHMENTS METADATA:\n" + "\n".join(metadata_lines)})
    return parts


def enhance_user_brief(task_id: str, brief: str, attachments: List[Attachment]) -> str:
    """
    Enhance the user-provided brief with default instructions to generate a realistic, production-ready app.
    Adds guidance about UI, accessibility, usability, code quality, and attachment handling.
    """
    enhanced = (
        f"TASK ID: {task_id}\n"
        f"USER BRIEF: {brief}\n\n"
        "Enhance the app generation as follows:\n"
        "- Produce a **modern, realistic, production-ready single-page app**.\n"
        "- Use clean, maintainable HTML, CSS, and JavaScript. Follow best practices.\n"
        "- Include responsive design and accessibility (ARIA attributes, semantic HTML).\n"
        "- Use CDNs for external libraries (Bootstrap, JS libraries, fonts).\n"
        "- Implement basic client-side validation where applicable.\n"
        "- Include inline comments explaining key parts of the code.\n"
        "- For attachments (images, CSV, JSON), automatically include fallbacks or references in the app.\n"
        "- README.md must clearly explain setup, usage, attachment references, and any assumptions.\n"
        "- LICENSE must be MIT.\n"
        "- Ensure the app looks and behaves like a **real web app**, not just a minimal demo.\n"
        "- Avoid placeholder content; populate the app with sample or meaningful data where appropriate.\n"
    )
    return enhanced




async def call_aipipe_for_files(task_id: str, brief: str, attachments: List[Attachment]) -> dict:
    """
    Call Aipipe/OpenRouter to generate index.html, README.md, LICENSE via function-calling tool.
    Returns dict with keys: index.html, README.md, LICENSE on success.
    Raises Exception on unrecoverable failure (caller may fallback).
    """
    from json import loads
    import asyncio
    import httpx

    # --- Configuration ---
    AIPIPE_API_KEY = settings.AIPIPE_API_KEY
    AIPIPE_BASE = settings.AIPIPE_BASE
    AIPIPE_MODEL = settings.AIPIPE_MODEL
    timeout_seconds = settings.AIPIPE_TIMEOUT_SECONDS

    if not AIPIPE_API_KEY:
        raise RuntimeError("AIPIPE_API_KEY not set")

    logger.info(f"[LLM] calling Aipipe for task {task_id} model={AIPIPE_MODEL}")

    # --- System Prompt ---
    system_prompt = (
        "You are an expert full-stack web engineer and UX designer. "
        "You MUST return exactly one function call to 'generate_files' with keys: "
        "index.html, README.md, LICENSE. No extra text. "
        "Files should be production-ready, deployable, and polished.\n\n"
        "Guidelines:\n"
        "- index.html: implement the brief, client-side only, responsive, accessible, semantic HTML.\n"
        "- Include inline comments explaining code logic.\n"
        "- Use CDN links for libraries (Bootstrap, JS, fonts) instead of local files.\n"
        "- Use attachments meaningfully: images should display, CSV/JSON should populate data.\n"
        "- README.md: clear setup, usage instructions, explanation of attachments, any assumptions.\n"
        "- LICENSE: must be MIT.\n"
        "- Ensure the app behaves like a real web app, with sample data where appropriate.\n"
        "- Avoid placeholders; populate content realistically."
    )

    # --- Helper: Enhance user brief ---
    def enhance_user_brief(task_id: str, brief: str, attachments: List[Attachment]) -> str:
        enhanced = (
            f"TASK ID: {task_id}\n"
            f"USER BRIEF: {brief}\n\n"
            "Enhancements for the app generation:\n"
            "- Produce a polished, production-ready single-page app.\n"
            "- Include responsive design and accessibility (ARIA, semantic HTML).\n"
            "- Validate inputs client-side where applicable.\n"
            "- Include inline comments explaining important code sections.\n"
            "- Reference attachments meaningfully in the app.\n"
            "- README.md must clearly explain setup, usage, and attachment usage.\n"
            "- LICENSE must be MIT.\n"
            "- App should appear and function as a real product, not minimal demo.\n"
        )
        return enhanced

    # --- Build user content ---
    enhanced_brief = enhance_user_brief(task_id, brief, attachments)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": enhanced_brief}
    ]

    # --- Attachments ---
    attachment_parts = await _build_attachment_parts_for_llm(attachments)
    for part in attachment_parts:
        if part.get("inlineData"):
            data_uri = f"data:{part['inlineData']['mimeType']};base64,{part['inlineData']['data']}"
            messages.append({"role": "user", "content": f"ATTACHMENT: {part['inlineData'].get('name','attachment')} -> {data_uri}"})
        elif part.get("type") == "text" and part.get("text"):
            messages.append({"role": "user", "content": part["text"]})

    # --- Initialize LLM Client ---
    try:
        client = OpenAIClient(api_key=AIPIPE_API_KEY, base_url=AIPIPE_BASE,
                              http_client=httpx.Client(timeout=timeout_seconds))
    except Exception as e:
        logger.exception(f"[LLM] client init failed: {e}")
        raise

    # --- Attempt retries ---
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=AIPIPE_MODEL,
                messages=messages,
                tools=[GENERATED_FILES_TOOL],
                tool_choice={"type": "function", "function": {"name": "generate_files"}},
                temperature=0.0,
                max_tokens=3000
            )

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls and tool_calls[0].function.name == "generate_files":
                generated = loads(tool_calls[0].function.arguments)
                # Validate output
                for k in ("index.html", "README.md", "LICENSE"):
                    if k not in generated or not isinstance(generated[k], str):
                        raise ValueError(f"Missing or invalid {k} in LLM response")
                logger.info(f"[LLM] Generated files for {task_id}")
                return generated
            else:
                raise ValueError("Model did not return expected function call 'generate_files'")

        except Exception as e:
            logger.warning(f"[LLM] attempt {attempt} error: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error("[LLM] all attempts failed, raising exception")
                raise












# -----------------------
# Git helpers (optional; only used if DRY_RUN is false and token provided)
# -----------------------
async def create_or_clone_repo(local_path: str, repo_name: str, repo_url_http: str, auth_url: str, round_index: int) -> git.Repo:
    safe_name = sanitize_repo_name(repo_name)
    if round_index == 1:
        # Try create repo via GitHub API if token present
        if not settings.GITHUB_TOKEN or not settings.GITHUB_USERNAME:
            logger.info("Skipping GitHub create: missing token or username")
        else:
            headers = {"Authorization": f"token {settings.GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
            payload = {"name": safe_name, "private": False, "auto_init": True}
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{settings.GITHUB_API_BASE}/user/repos", json=payload, headers=headers)
                # read body
                body = None
                try:
                    body = r.json()
                except Exception:
                    body = r.text
                if r.status_code == 201:
                    logger.info(f"Created GitHub repo {safe_name}")
                elif r.status_code == 422:
                    logger.warning(f"Create returned 422: {body}")
                elif r.status_code in (401,403):
                    logger.warning(f"Auth issue creating repo: status {r.status_code}, body: {body}")
                else:
                    logger.warning(f"Unexpected status creating repo: {r.status_code}, body: {body}")

    # If remote exists or on subsequent rounds, try to clone; otherwise init local repo
    try:
        repo = await asyncio.to_thread(git.Repo.init, local_path)
        repo.create_remote('origin', auth_url)
        logger.info(f"Initialized local repo at {local_path}")
        return repo
    except Exception as e:
        logger.warning(f"Failed to init local repo: {e}, trying clone")
        try:
            repo = await asyncio.to_thread(git.Repo.clone_from, auth_url, local_path)
            logger.info("Cloned repo")
            return repo
        except Exception as e2:
            logger.exception(f"Clone failed: {e2}")
            raise

async def commit_push(repo: git.Repo, commit_msg: str):
    try:
        await asyncio.to_thread(repo.git.add, A=True)
        commit = await asyncio.to_thread(lambda m: repo.index.commit(m), commit_msg)
        await asyncio.to_thread(lambda: repo.git.branch('-M', 'main'))
        await asyncio.to_thread(lambda: repo.git.push('--set-upstream', 'origin', 'main', force=True))
        logger.info("Committed and pushed to origin main")
        return getattr(commit, "hexsha", "")
    except Exception as e:
        logger.exception(f"Commit/push failed: {e}")
        raise

# -----------------------
# Background orchestration
# -----------------------
async def generate_and_deploy(task: TaskRequest):
    acquired = False
    try:
        await task_semaphore.acquire()
        acquired = True
        logger.info(f"Start task {task.task} round {task.round}")
        task_id = task.task
        repo_name = sanitize_repo_name(task.task)
        base_dir = os.path.join(os.getcwd(), "generated_tasks")
        local_path = os.path.join(base_dir, repo_name)

        # cleanup
        if os.path.exists(local_path):
            try:
                shutil.rmtree(local_path)
            except Exception:
                pass
        safe_makedirs(local_path)

        # Save attachments
        saved = await save_attachments_locally(local_path, task.attachments)

        # Generate files: prefer LLM when DRY_RUN is disabled
        files = None
        if not settings.DRY_RUN:
            try:
                files = await call_aipipe_for_files(task_id, task.brief, task.attachments)
            except Exception as e:
                logger.error(f"[LLM] generation failed for {task_id}: {e}")
                raise  # don't silently fallback; let user fix API key / network
        if not files:
            files = local_generate_files(task_id, task.brief, task.attachments)



        if not files:
            # fallback to local generator
            files = local_generate_files(task_id, task.brief, task.attachments)

        # Save files
        for fname, content in files.items():
            write_file(os.path.join(local_path, fname), content)
            logger.info(f"Wrote generated file: {fname}")


        # If attachments include images, ensure sample fallback exists (already saved)
        # If DRY_RUN, skip GitHub push and notify with fake repo/pages URLs
        if settings.DRY_RUN:
            logger.info("DRY_RUN is True: skipping GitHub push")
            repo_url = f"https://github.com/{settings.GITHUB_USERNAME or 'local'}/{repo_name}"
            commit_sha = "deadbeef"
            pages_url = f"{settings.GITHUB_PAGES_BASE or 'http://localhost:8000'}/{repo_name}/"
            payload = {"email": task.email, "task": task.task, "round": task.round, "nonce": task.nonce,
                       "repo_url": repo_url, "commit_sha": commit_sha, "pages_url": pages_url}
            await notify_evaluator(task.evaluation_url, payload)
            return
        
        # else — attempt to create local repo and push (will require git + proper token/remote)
        auth_url = f"https://{settings.GITHUB_USERNAME}:{settings.GITHUB_TOKEN}@github.com/{settings.GITHUB_USERNAME}/{repo_name}.git"
        http_remote = f"https://github.com/{settings.GITHUB_USERNAME}/{repo_name}.git"
        repo = await create_or_clone_repo(local_path, repo_name, http_remote, auth_url, task.round)
        commit_sha = await commit_push(repo, f"Task {task.task} round {task.round}")
        pages_url = await enable_github_pages(repo_name)
        if not pages_url:
            pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/" if settings.GITHUB_PAGES_BASE else ""
        repo_url = f"https://github.com/{settings.GITHUB_USERNAME}/{repo_name}"
        payload = {"email": task.email, "task": task.task, "round": task.round, "nonce": task.nonce,
                   "repo_url": repo_url, "commit_sha": commit_sha, "pages_url": pages_url}
        await notify_evaluator(task.evaluation_url, payload)

    except Exception as e:
        logger.exception(f"Task failed: {e}")
    finally:
        if acquired:
            task_semaphore.release()
        flush_logs()

async def enable_github_pages(repo_name: str):
    if not settings.GITHUB_TOKEN or not settings.GITHUB_USERNAME:
        logger.warning("Skipping GitHub Pages: missing credentials")
        return None
    url = f"{settings.GITHUB_API_BASE}/repos/{settings.GITHUB_USERNAME}/{repo_name}/pages"
    headers = {
        "Authorization": f"token {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    body = {"source": {"branch": "main", "path": "/"}}
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            r = await client.post(url, headers=headers, json=body)
            if r.status_code in [201, 202]:
                pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/"
                logger.info(f"GitHub Pages enabled at {pages_url}")
                return pages_url
            else:
                logger.warning(f"GitHub Pages activation failed: {r.status_code} {r.text}")
        except Exception as e:
            logger.warning(f"GitHub Pages activation error: {e}")
    return None



def _task_done_callback(t: asyncio.Task):
    try:
        exc = t.exception()
        if exc:
            logger.error(f"Background task exception: {exc}")
        else:
            logger.info("Background task finished")
    except asyncio.CancelledError:
        logger.warning("Background task cancelled")
    finally:
        flush_logs()

# -----------------------
# Endpoints
# -----------------------
@app.post("/ready")
async def receive_task(task_data: TaskRequest, background_tasks: BackgroundTasks, request: Request):
    global last_received_task, background_tasks_list
    if not verify_secret(task_data.secret):
        logger.warning("Unauthorized attempt")
        raise HTTPException(status_code=401, detail="Unauthorized")
    last_received_task = {"task": task_data.task, "round": task_data.round, "brief": (task_data.brief[:250] + "...") if len(task_data.brief) > 250 else task_data.brief, "time": datetime.utcnow().isoformat()+"Z"}
    bg = asyncio.create_task(generate_and_deploy(task_data))
    bg.add_done_callback(_task_done_callback)
    background_tasks_list.append(bg)
    background_tasks.add_task(lambda: None)
    logger.info(f"Received task {task_data.task}")
    flush_logs()
    return JSONResponse(status_code=200, content={"status":"processing_scheduled","task":task_data.task})

@app.post("/evaluation-receiver")
async def evaluation_receiver(req: Request):
    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    # For simplicity, accept and log
    logger.info("Evaluator callback received:")
    logger.info(json.dumps(payload, indent=2))
    flush_logs()
    return {"status":"ok","received": True}

@app.get("/status")
async def status():
    return {"last_received_task": last_received_task, "running_background_tasks": len([t for t in background_tasks_list if not t.done()])}

@app.get("/health")
async def health():
    return {"status":"ok","timestamp": datetime.utcnow().isoformat()+"Z"}

@app.get("/logs")
async def get_logs(lines: int = Query(200, ge=1, le=5000)):
    path = settings.LOG_FILE_PATH
    if not os.path.exists(path):
        return PlainTextResponse("Log file not found.", status_code=404)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            buf = bytearray()
            block = 1024
            while size > 0 and len(buf) < lines * 2000:
                read_size = min(block, size)
                f.seek(size - read_size)
                buf.extend(f.read(read_size))
                size -= read_size
            text = buf.decode(errors="ignore").splitlines()
            last_lines = "\n".join(text[-lines:])
            return PlainTextResponse(last_lines)
    except Exception as e:
        logger.exception(f"Logs read failed: {e}")
        return PlainTextResponse(f"Error: {e}", status_code=500)

# startup keepalive (useful to see activity)
@app.on_event("startup")
async def startup_event():
    async def keepalive():
        while True:
            try:
                logger.info("[KEEPALIVE] heartbeat")
                flush_logs()
            except Exception:
                pass
            await asyncio.sleep(30)
    asyncio.create_task(keepalive())
