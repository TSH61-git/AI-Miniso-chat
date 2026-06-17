# chat_service.py
import numpy as np
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

# ── SEARCH ENDPOINT ──────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str       # מה שהמשתמש חיפש (למשל: "בגדים חמים לחורף")
    products: list   # רשימת המוצרים שתגיע אלינו מה-.NET
    top_k: int = 5   # כמה תוצאות הכי קרובות להחזיר (ברירת מחדל: 5)

# ── זיכרון מטמון גלובלי לאימבדינגס של המוצרים ─────────────────
# המפתח יהיה שם המוצר (או מזהה ייחודי), והערך יהיה רשימת המספרים (האימבדינג)
embeddings_cache = {}

@app.post('/search')
async def search(req: SearchRequest):
    if not req.products:
        return {'results': []}

    query_embedding = get_embedding(req.query)

    scored = []
    for p in req.products:
        product_name = p.get('productName', '')
        product_text = f"{product_name} {product_name} {p.get('description','')}"
        
        # 2. בדיקה: האם כבר חישבנו בעבר אימבדינג עבור המוצר הזה?
        if product_name in embeddings_cache:
            # אם כן - שולפים מהזיכרון המקומי בשבריר שנייה בלי לפנות ל-OpenAI!
            product_embedding = embeddings_cache[product_name]
        else:
            # אם לא (למשל מוצר חדש או הרצה ראשונה) - מחשבים ושומרים במטמון לפעמים הבאות
            product_embedding = get_embedding(product_text)
            embeddings_cache[product_name] = product_embedding
        
        # 3. חישוב הדמיון
        score = cosine_similarity(query_embedding, product_embedding)
        scored.append({**p, 'score': round(score, 3)})

    # מיון המוצרים מהציון הגבוה לנמוך
    results = sorted(scored, key=lambda x: x['score'], reverse=True)
    
    # תיקון: מחזירים רק מוצרים שציון ההתאמה שלהם גבוה מ-0.52 (או 0.5) כדי לנפות "רעשים"
    filtered_results = [p for p in results if p['score'] >= 0.42]
    
    # מחזירים את המוצרים המנופים (עד top_k מוצרים)
    return {'results': filtered_results[:req.top_k]}

# ── EMBEDDING FUNCTIONS ──────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    # פנייה ל-OpenAI כדי לתרגם את הטקסט לרשימה של כ-1500 מספרים המייצגים משמעות
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    # נוסחה מתמטית המשווה בין שתי רשימות המספרים ומחזירה ציון קירבה בין 0 ל-1
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))