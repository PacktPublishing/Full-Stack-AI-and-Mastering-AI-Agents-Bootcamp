import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "mistral", "prompt": "What is AI?", "stream": False}
)

print(response.json()["response"])