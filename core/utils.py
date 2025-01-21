from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
parse = client.beta.chat.completions.parse
create = client.chat.completions.create

def systemp(value: str) -> str:
    return {"role": "system", "content": value}
def userp(value: str) -> str:
    return {"role": "user", "content": value}
def assistantp(value: str) -> str:
    return {"role": "assistant", "content": value}
