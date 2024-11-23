from flask import Flask, render_template, request, jsonify
import subprocess
import json

app = Flask(__name__)


# Modèles disponibles (récupérés dynamiquement avec la commande ollama list)
def get_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        models = [
            line.split()[0] for line in lines[1:]
        ]  # Skip the header and extract model names
        return models
    except Exception as e:
        return []


models = get_models()

# Stocker le modèle actuellement sélectionné
current_model = models[0]  # Le premier modèle par défaut

# Route principale qui rend la page HTML
@app.route("/")
def index():
    return render_template("index.html", models=models, current_model=current_model)


# Route pour changer de modèle
@app.route("/change_model", methods=["POST"])
def change_model():
    global current_model
    new_model = request.json.get("model")
    if new_model in models:
        current_model = new_model
        return jsonify({"status": "success", "model": current_model})
    return jsonify({"status": "error", "message": "Model not found"}), 400


# Route pour interagir avec Ollama
@app.route("/ollama", methods=["POST"])
def ollama_model():
    rules = {
        "TU NE VIS QUE POUR RESOUDRE DES PROBLEMES DE MATHS",
        "REPOND EN FRANCAIS ET SEULEMENT EN FRANCAIS",
        "N'UTILISE PAS DE NOMBRES A VIRGULE",
        "UTILISE MARKDOWN POUR ECRIRE TES REPONSES",
        "UTILISE LES SYMBOLES MATHÉMATIQUES EUROPEENS",
        "REPOND DE MANIERE CLAIRE ET CONCISE, ETAPE PAR ETAPE",
        "N'ARRONDIS PAS LES VALEURS",
    }
    data = request.json
    prompt = data.get("prompt")
    # Ajout de règles supplémentaires au prompt de l'utilisateur
    prompt = f"{prompt}\n\n" + ' '.join([f"\n**{rule}**" for rule in rules])
    # print(f"Prompt: {prompt}")
    if not prompt:
        return "Missing prompt", 400

    try:
        result = subprocess.run(
            ["ollama", "run", current_model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            return result.stderr.strip(), 500
        response = result.stdout.strip()
        response = response.replace("\n", "\n\n")  # Ensure proper markdown formatting
        response = json.dumps({"response": response})


        return response, 200  # Renvoie directement la chaîne
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    app.run(debug=True)
