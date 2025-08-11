# api.py
# Endpunkt-Definitionen f�r das Movie-Recommender-System
from flask import Flask
app = Flask(__name__)
@app.get("/health")
def health():
    return {"ok": True}
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
