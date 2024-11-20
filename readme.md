# Super AI Maths Solver

Super AI Maths Solver is a web application that allows users to interact with an AI model to solve math problems. The application supports multiple languages and provides clear, step-by-step solutions using European mathematical symbols.

## Features

- Interact with AI to solve math problems
- Supports multiple languages for responses
- Dynamic model selection
- Clear and concise step-by-step solutions
- Uses Markdown for formatting responses
- Supports MathJax for rendering mathematical symbols

## Prerequisites

- Python 3.x
- Flask
- Googletrans
- Ollama (for managing AI models)

## Installation

1. Clone the repository:

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