import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

assistant = pc.assistant.Assistant(assistant_name="help-scout")

msg = Message(content="How old is the earth?")
resp = assistant.chat(messages=[msg])

print(resp["message"]["content"])
