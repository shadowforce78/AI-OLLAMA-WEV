# Super AI Maths Solver

Super AI Maths Solver est une application web qui permet aux utilisateurs d'interagir avec un modèle d'IA pour résoudre des problèmes de mathématiques. L'application prend en charge plusieurs langues et fournit des solutions claires et étape par étape en utilisant des symboles mathématiques européens.

## Fonctionnalités

- Interagir avec l'IA pour résoudre des problèmes de mathématiques
- Prend en charge plusieurs langues pour les réponses
- Sélection dynamique de modèles
- Solutions claires et concises étape par étape
- Utilise Markdown pour formater les réponses
- Prend en charge MathJax pour le rendu des symboles mathématiques

## Prérequis

- Python 3.x
- Flask
- Googletrans
- Ollama (pour gérer les modèles d'IA)

## Installation

1. Clonez le dépôt :

   ```sh
   git clone https://github.com/shadowforce78/AI-OLLAMA-WEV.git
   cd AI-OLLAMA-WEV
   ```

2. Install the required Python packages:

   ```sh
   pip install -r requirements.txt
   ```

3. Install Ollama:

   Follow the instructions on the [Ollama website](https://ollama.com) to install Ollama on your system.

## Usage

1. Start the Flask server:

   ```sh
   python server.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Interact with the AI by selecting a model, choosing a language, and entering your math problem.

4. Pull the required model for better usage:

   ```sh
   ollama pull mathstral
   ```

## Project Structure

- `templates/index.html`: The main HTML template for the web application.
- `server.py`: The Flask server handling API requests and interactions with Ollama.

## API Endpoints

- `/`: Renders the main HTML page.
- `/change_model`: Changes the currently selected AI model.
- `/ollama`: Sends a prompt to the AI model and returns the response.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [Googletrans](https://py-googletrans.readthedocs.io/en/latest/)
- [Ollama](https://ollama.com)
- [MathJax](https://www.mathjax.org/)
