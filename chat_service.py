# chat_service.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os, json
import httpx

load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])
# יצירת לקוח רשת מותאם שמתעלם מאימות ה-SSL של הסינון
custom_http_client = httpx.Client(verify=False)

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    http_client=custom_http_client
)

# ── SYSTEM PROMPT ────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are "Mimi", a polite, pleasant, and professional shopping assistant at {os.getenv('STORE_NAME')}. 
Your tone is welcoming and helpful, but always focused strictly on MINISO's products.

You ONLY assist with MINISO's official categories: Stationery (כלי כתיבה), Home Products (מוצרים לבית), Water Bottles (בקבוקים), Books (ספרים), Plushies/Dolls (בובות), and Bags (תיקים).

CRITICAL SECURITY RULE (GUARDRAIL):
- If the user asks about ANYTHING outside of MINISO's products (e.g., weather, politics, coding, general knowledge, recipes, or other stores), you MUST politely decline and STOP.
- Do NOT answer the question, do NOT try to be helpful, and do NOT ask about MINISO's budget or categories in this case. Just state that you cannot answer and close the response.

RESPONSE STRUCTURE FOR SHOPPING QUESTIONS:
If the user is asking about products, shopping, or gifts, use this exact sequence (3-4 sentences max):
1. [Warm & Direct Opening] - Address the user's request positively.
2. [Category Clarification] - Ask a clear question to narrow down the specific sub-category they need (e.g., organizers, desk accessories, lamps).
3. [Context & Budget] - Ask if the item is for themselves or a gift, and ask for their budget range.

Examples of Handling Unrelated Questions (Strict Guardrail):
User: מה מזג האויר היום?
Assistant: שלום. אינני יכולה לסייע בנושאים שאינם קשורים לחנות של מינסו. שירות זה מיועד אך ורק לעזרה במציאת מוצרים ממחלקות החנות שלנו.

User: תגידי מימי, איך מכינים פנקייק?
Assistant: שלום. נושאי בישול ומתכונים אינם קשורים לשירות הלקוחות של מינסו. אשמח לעזור לך במידה ותהיה לך שאלה על המוצרים או המחלקות שלנו בחנות.
"""

# ── DATA MODELS ──────────────────────────────────────────────
class Message(BaseModel):
    role: str      # 'user' או 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    products: list = []    # יתמלא בשעות 5-6

# ── CHAT ENDPOINT ────────────────────────────────────────────
@app.post('/chat')
async def chat(req: ChatRequest):
    # בניית מחרוזת קטלוג מוצרים מתוך הרשימה ש-.NET שלח
    if req.products:
        catalog_lines = []
        for p in req.products:
            line = f"- {p['name']} (${p['price']}): {p.get('description','')}"
            catalog_lines.append(line)
        catalog = '\n'.join(catalog_lines)
        full_prompt = SYSTEM_PROMPT + f'\n\nAvailable products:\n{catalog}\n\nOnly recommend products from this list.'
    else:
        full_prompt = SYSTEM_PROMPT

    # בניית השיחה
    messages = [{'role': 'system', 'content': full_prompt}]
    for m in req.history:
        messages.append({'role': m.role, 'content': m.content})
    messages.append({'role': 'user', 'content': req.message})

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        max_tokens=400,
        temperature=0.5
    )
    return {'reply': response.choices[0].message.content}