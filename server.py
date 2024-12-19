from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Obtener rutas absolutas para los modelos
saved_model_legal_path = os.path.abspath("./saved_model_legal")
saved_models_article_path = os.path.abspath("./saved_model_article")

# Cargar modelos y tokenizador desde rutas absolutas
model_legal = BertForSequenceClassification.from_pretrained(
    saved_model_legal_path
).to(device)

model_article = BertForSequenceClassification.from_pretrained(
    saved_models_article_path
).to(device)

tokenizer = BertTokenizer.from_pretrained(saved_model_legal_path)

# Mapeo de artículos
unique_articles = ["1101", "1124", "1256", "1454", "1484", "1504", "1537", "1902", "1903", "1911"]  # Ejemplo, actualiza con tu lista
article_to_idx = {article: idx for idx, article in enumerate(unique_articles)}
idx_to_article = {idx: article for article, idx in article_to_idx.items()}

# Crear la aplicación Flask
app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()  # Leer el JSON enviado en la solicitud
    clauses = data.get("clauses", [])
    results = []

    for clause in clauses:
        # Tokenizar la cláusula
        inputs = tokenizer(clause, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

        # Predicción legal/ilegal
        model_legal.eval()
        with torch.no_grad():
            outputs_legal = model_legal(**inputs)
            predicted_legal = torch.argmax(outputs_legal.logits).item()

        # Si es ilegal, predecir el artículo
        if predicted_legal == 1:
            model_article.eval()
            with torch.no_grad():
                outputs_article = model_article(**inputs)
                predicted_article_idx = torch.argmax(outputs_article.logits).item()
                predicted_article = idx_to_article.get(predicted_article_idx, "Unknown")
            results.append({"clause": clause, "violation": True, "article": predicted_article})
        else:
            results.append({"clause": clause, "violation": False})

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5320)
    print("Server is running on http://0.0.0.0:5320")
