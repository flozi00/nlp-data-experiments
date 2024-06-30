import datasets
from TOKENS import BOT, PROMPTER, END, SYSTEM
import random
from tqdm import tqdm

SYSTEM_PROMPTS = [
    "Gegeben ist ein Text und eine Frage. Die Antwort auf die Frage ist eine der vorgegebenen Antworten.",
    "Im folgenden Text ist eine Frage gestellt. Die Antwort auf die Frage ist eine der gegebenen Antworten.",
    "Im folgenden wird eine passende Antwortmöglichkeit zu einer Frage gesucht.",
    "Gegeben ist eine Konversation zwischen einem Assistenten und einem Nutzer. Der Nutzer stellt eine Frage und der Assistent antwortet. Die Antwort des Assistenten ist eine der vorgegebenen Antworten.",
    "Im folgenden beantwortet ein Assistent eine Frage des Nutzers. Die Antwort des Assistenten ist eine der Antworten welche der Nutzer aufzählt.",
    "Im folgenden sucht der Nutzer die passende Antwort zu einer Frage. Der Nutzer hat die Auswahl zwischen vier Antworten. Der Assistent sucht dem Nutzer die passende Antwort aus.",
]


def belebele() -> tuple[list, list, list]:
    prompt = []
    chosen = []
    rejected = []
    ds = datasets.load_dataset("facebook/belebele", split="deu_Latn")

    for entry in tqdm(ds, desc="belebele"):
        flores_passage = entry["flores_passage"]
        question = entry["question"]
        mc_answer1 = "1. " + entry["mc_answer1"]
        mc_answer2 = "2. " + entry["mc_answer2"]
        mc_answer3 = "3. " + entry["mc_answer3"]
        mc_answer4 = "4. " + entry["mc_answer4"]

        if "1" in entry["correct_answer_num"]:
            correct_answer = mc_answer1
        elif "2" in entry["correct_answer_num"]:
            correct_answer = mc_answer2
        elif "3" in entry["correct_answer_num"]:
            correct_answer = mc_answer3
        elif "4" in entry["correct_answer_num"]:
            correct_answer = mc_answer4

        _PROMPT = f"""{random.choice(SYSTEM_PROMPTS)}

{mc_answer1}
{mc_answer2}
{mc_answer3}
{mc_answer4}

Text: {flores_passage}
Frage: {question}"""

        for x in [mc_answer1, mc_answer2, mc_answer3, mc_answer4]:
            if x == correct_answer:
                continue
            else:
                prompt.append(_PROMPT)
                chosen.append(correct_answer)
                rejected.append(x)

    return prompt, chosen, rejected
