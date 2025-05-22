import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
MODEL = "gpt-4o-mini"


# Helper to read & encode image
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_b64 = encode_image("test.jpg")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are an AI assistant that can analyze images. Your job is to based on the input images and then anaylze the surronding enviroment."}, 
        {"role": "user", "content": [
            {"type": "text", "text": "Hello, how are you?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{image_b64}"
            }}
        ]}
    ],
    temperature=0.0
)

print(response.choices[0].message.content)
