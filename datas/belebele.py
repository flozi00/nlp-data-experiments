import datasets
from TOKENS import BOT, PROMPTER, END


def belebele():
    all_rows = []
    all_labels = []
    from_ds = []
    ds = datasets.load_dataset(
        "facebook/belebele",
        "deu_Latn",
        split="test",
    )

    for entry in ds:
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

        PROMPT = f"""{PROMPTER}Gegeben ist ein Text in deutscher Sprache und eine Frage auf Deutsch.
Die Antwort auf die Frage ist eine der vier vorgegebenen Antworten.
Antwortmöglichkeiten:
{mc_answer1}
{mc_answer2}
{mc_answer3}
{mc_answer4}
Text: {flores_passage}
Frage: {question}{END}{BOT}{correct_answer}{END}"""

        all_rows.append(PROMPT)
        all_labels.append("closed_qa")
        from_ds.append("facebook/belebele")

    return all_rows, all_labels, from_ds
