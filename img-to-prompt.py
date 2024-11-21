from PIL import Image
import pytesseract
import re

def extract_text_with_math(image_path):
    # Extraire le texte brut avec pytesseract
    image = Image.open(image_path)
    raw_text = pytesseract.image_to_string(image=image, lang='fra')

    # Transformer les formules mathématiques détectées en Markdown
    processed_text = convert_to_markdown(raw_text)

    return processed_text

def convert_to_markdown(text):
    # Exemple simple de transformation de fractions, exposants, etc.
    # (À ajuster selon les cas d'usage spécifiques)
    
    # Exemple : fractions (1/2) => Markdown inline math $\\frac{1}{2}$
    text = re.sub(r'(\d+)/(\d+)', r'$\frac{\1}{\2}$', text)
    
    # Exemple : exposants (x^2) => Markdown inline math $x^2$
    text = re.sub(r'(\w+)\^(\d+)', r'$\1^\2$', text)
    
    # Exemple : remplace des parties complexes détectées comme "mathématiques"
    # Les expressions entre parenthèses "( ... )" dans un contexte probable de math
    text = re.sub(r'\((.*?)\)', r'$\1$', text)

    return text

# Exemple d'utilisation
if __name__ == "__main__":
    markdown_text = extract_text_with_math("image.png")
    print(markdown_text)
