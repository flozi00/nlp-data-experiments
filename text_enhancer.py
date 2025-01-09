SYS_PROMPT = """Deine Aufgabe ist es, Texte zu bearbeiten, die kein gutes Format haben, und sie in ein gut lesbares Format für den schulischen Bereich zu schreiben. Die Texte müssen detailliert, aber leicht verständlich und lehrreich verfasst sein.

**Verhaltensweisen und Regeln:**

1. **Textauswertung:**

   a) Bitte den Benutzer, den Text zur Verfügung zu stellen, der formatiert werden soll.

   b) Analysiere den Text sorgfältig, um die wichtigsten Informationen und Kernaussagen zu identifizieren.

   c) Achte auf die Zielgruppe (Schüler) und passe den Schreibstil und die Komplexität entsprechend an.


2. **Formatierung:**

   a) Gliedere den Text in übersichtliche Abschnitte mit aussagekräftigen Überschriften.

   b) Verwende Aufzählungen, Tabellen oder Grafiken, um Informationen klar und prägnant darzustellen.

   c) Achte auf eine korrekte Rechtschreibung, Grammatik und Zeichensetzung.

   d) Stelle sicher, dass der formatierte Text den schulischen Anforderungen entspricht.


3. **Zusätzliche Hinweise:**

   a) Erkläre Fachbegriffe und komplexe Konzepte in einfachen Worten.

   b) Verwende Beispiele und Illustrationen, um das Verständnis zu erleichtern.

   c) Füge gegebenenfalls zusätzliche Informationen hinzu, um den Text lehrreicher zu gestalten.


**Gesamter Ton:**
    * Sei objektiv und sachlich.

    * Verwende eine klare und präzise Sprache.

    * Vermeide Umgangssprache oder Slang.

    * Gestalte den Text ansprechend und motivierend für Schüler."""


import hashlib
import json
import os

import datasets
from openai import OpenAI

try:
    os.mkdir("logging_enhancements")
except FileExistsError:
    pass

client = OpenAI(
    base_url=os.getenv("OAI_BASE_URL"),
    api_key=os.getenv("OAI_KEY"),
)

MODEL_TO_USE = "meta-llama/Llama-3.3-70B-Instruct"

ds = datasets.load_dataset("flozi00/Fineweb2-German-Eduscore-4andMore", split="train")


def enhance_text(text: str):
    # Generate hash for caching
    msg_hash = hashlib.md5(text.encode()).hexdigest()

    # Check cache
    if os.path.exists(f"logging_enhancements/{msg_hash}.json"):
        with open(f"logging_enhancements/{msg_hash}.json", "r") as f:
            log = json.load(f)
            return log["enhanced_text"]

    # Prepare message for API
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text},
    ]

    # Get completion
    completion = client.chat.completions.create(
        model=MODEL_TO_USE,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
    )

    enhanced_text = completion.choices[0].message.content

    # Cache result
    with open(f"logging_enhancements/{msg_hash}.json", "w+") as f:
        json.dump(
            {"original_text": text, "enhanced_text": enhanced_text},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return enhanced_text


# Example usage:
for data in ds:
    enhanced = enhance_text(data["text"])
    print("Original:", data["text"])
    print("\nEnhanced:", enhanced)
    print("-" * 80)
    break
