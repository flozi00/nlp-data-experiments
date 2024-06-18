from openai import OpenAI
import datasets
from httpx import URL
import os
import json

from tqdm import tqdm


EDU_PROMPT = """Nachfolgend findest du einen Auszug aus einer Webseite. Beurteile, ob die Seite einen hohen pädagogischen Wert hat und in einem pädagogischen Umfeld für den Unterricht von der Grundschule bis zur Universität nützlich sein könnte, indem du das unten beschriebene 5-Punkte-Bewertungssystem anwendest. Die Punkte werden auf der Grundlage der Erfüllung der am besten passenden Kriterien gewählt:

- 0 Punkte: Der Inhalt ist nicht organisiert und schwer zu lesen. Der Text enthält Werbung oder irrelevante Informationen zum lehren von Inhalten. Der Text ist nicht neutral sondern enthält persöhnliche Sichtweisen. Beispiel: Tweets, Chatnachrichten oder Forenbeiträge.
- 1 Punkt: Der Text ist für den privaten Gebrauch bestimmt und enthält Werbung oder irrelevante Informationen. Der Text ist nicht neutral und spiegelt zum Teil persönliche Sichtweisen wider. Beispiel: Ein Blogbeitrag, der hauptsächlich auf persönliche Erfahrungen eingeht und nur gelegentlich nützliche Informationen bietet.
- 2 Punkte: Der Text ist neutral geschrieben, aber enthält Werbung oder irrelevante Informationen. Die enthaltenen Informationen können zeitlich vergänglich sein. Beispiel: Ein Artikel oder Nachrichtenbeitrag.
- 3 Punkte: Der Text enthält viele Informationen und ist leicht verständlich. Der Text ist neutral geschrieben und enthält keine Werbung oder irrelevante Informationen. Beispiel: Ein Wikipedia-Artikel.
- 4 Punkte: Der Text ist neutral geschrieben und enthält keine Werbung oder irrelevante Informationen. Der Text enthält tiefergehendes Wissen und ist für den Unterricht von der Grundschule bis zur Universität nützlich. Beispiel: Ein wissenschaftlicher Artikel oder ein Lehrbuch
- 5 Punkte: Der Text beeinhaltet tiefergehendes Wissen, ist dabei aber dennoch leicht verständlich, sodass jeder daraus lernen und sich neue Fähigkeiten aneignen kann. Beispielsweise Schritt für Schritt Anleitungen, Erklärungen oder Definitionen.

Nachdem du den Auszug geprüft hast: 
- Wähle eine Punktzahl von 0 bis 5, die am besten beschreibt, wie nützlich der Inhalt für den Unterricht von der Grundschule bis zur Universität ist.
- Begründe kurz deine ausgewählte Punktzahl, bis zu 100 Wörter.
- Antworte im folgenden Format "<Gesamtpunktzahl> : <Begründung>"
"""

LOKAL = URL("http://127.0.0.1:1338/v1/")
MODEL = "pL-Community/GermanEduScorer-Qwen2-1.5b"

client = OpenAI(
    base_url=LOKAL,
    api_key="any_key",
)

messages = [
    {"role": "system", "content": EDU_PROMPT},
    {
        "role": "user",
        "content": "Das Landhaus Bautzner Straße 99 ist ein 1857 erbautes Wohnhaus im Preußischen Viertel in Dresden. Das für den Apotheker Ernst Ludwig Opitz erbaute villenartige Gebäude steht heute unter Denkmalschutz. Das Haus befindet sich kurz vor der Kreuzung mit der Stolpener Straße, sein Gartengrundstück grenzt an die Radeberger Straße. Beschreibung Das zweigeschossige Gebäude mit teilweise ausgebautem Dachgeschoss verfügt über einen Mittelrisaliten mit einem relativ flachen Zwerchhaus über dem verkröpften Dachgesims. Der Risalit nimmt etwa ein Drittel der Gebäudebreite ein und zeigt im Erdgeschoss und ersten Obergeschoss eine Rundbogentür zwischen zwei Rundbogenfenstern. Im ersten Obergeschoss gehen Tür und Fenster auf einen Balkon, im Erdgeschoss auf eine in den Vorgarten führende niedrige Treppe. Das Erdgeschoss ist in Sandstein ausgeführt, ebenso wie alle Fenstergewände und die Verdachung der Fenster im Obergeschoss. Auf der verputzten Gartenseite befindet sich ein nur einachsiger, schmalerer Risalit, der ebenfalls in ein Zwerchhaus übergeht. Links daneben befindet sich ein über beide Fensterachsen gehender zweigeschossiger hölzerner Wintergarten. Geschichte Das Gebäude gehört zur ersten Bauphase des Preußischen Viertels entlang der heutigen Bautzner Straße. Die Erschließung des Viertels durch Straßen begann erst nach 1860. Der Stadtplan von 1849 zeigt neben dem südlich der heutigen Bautzener Straße gelegenen Linckeschen Bad und einer Kaffeefabrik am östlichen Prießnitzufer noch keine Bebauung. Lediglich die „Straße nach Bautzen“ und die Radeberger Straße waren als Landstraßen bereits vorhanden. Zur Zeit der Erbauung hieß dieser Teil der Bautzner Straße Schillerstraße, das Landhaus hatte die Adresse Schillerstraße 4. Schon seit der Erbauung wurden der erste Stock und die Dachwohnung vermietet. Im Jahr 1861 bewohnte der Bauherr Ernst Ludwig Opitz das Parterre des Gebäudes. Im ersten Obergeschoss lebte die Witwe Therese Schweighofer, in der Dachwohnung darüber der im Adressbuch als Restaurateur bezeichnete C. Hopfe. Seit etwa 1995 stand das Gebäude leer, der Garten verwilderte. An der Bausubstanz entstanden durch den langen Leerstand, unzureichende Sicherung, eindringendes Wasser und einen Brand im Obergeschoss gravierende Schäden. Im April 2010 wurde das in der Denkmalliste geführte Gebäude verkauft und saniert und wird jetzt als Kunstauktionshaus für SCHMIDT Kunstauktionen Dresden, genutzt. Einzelnachweise Kulturdenkmal in Dresden Bautzner Strasse Erbaut in den 1850er Jahren Bautzner Straße Denkmalgeschütztes Bauwerk in Dresden Radeberger Vorstadt",
    },
]

response = client.chat.completions.create(
    messages=messages,
    temperature=0.01,
    model=MODEL,
    max_tokens=4,
)
print(response.choices[0].message.content)

CACHED_RATINGS = {}

if os.path.exists("cached_rating.json") is False:
    with open("cached_rating.json", "w") as f:
        f.write("{}")
        CACHED_RATINGS = {}
else:
    with open("cached_rating_backup.json", "r") as f:
        CACHED_RATINGS: dict = json.loads(f.read())
    with open("cached_rating.json", "r") as f:
        CACHED_RATINGS.update(json.loads(f.read()))


def gen_from_glot():
    if os.path.exists("glot_ds_cache") is True:
        return datasets.load_from_disk("glot_ds_cache")

    glot_ds = datasets.load_dataset(
        "cis-lmu/GlotCC-V1", "deu-Latn", split="train", streaming=True
    )
    ds_dict = {"text": [], "source": []}
    LIMIT = 890_000

    for entry in tqdm(glot_ds):
        ds_dict["text"].append(entry["content"])
        ds_dict["source"].append(entry["warc-target-uri"])
        if len(ds_dict["text"]) > LIMIT:
            break

    g_ds = datasets.Dataset.from_dict(ds_dict)
    g_ds.save_to_disk("glot_ds_cache")

    return g_ds


def get_score(text):
    messages = [
        {"role": "system", "content": EDU_PROMPT},
        {"role": "user", "content": text},
    ]
    response = client.chat.completions.create(
        messages=messages,
        temperature=0.01,
        model=MODEL,
        max_tokens=2,
    )
    score = response.choices[0].message.content
    for x in score:
        try:
            score = int(x)
            if score > 5:
                score = ""
        except ValueError:
            continue
    if isinstance(score, int) is False:
        return ""
    else:
        return str(score)


def map_scorer(example):
    global CACHED_RATINGS
    if example["text"] in CACHED_RATINGS.keys():
        score = CACHED_RATINGS[example["text"]]
    else:
        score = get_score(example["text"])
        CACHED_RATINGS[example["text"]] = score

        if len(CACHED_RATINGS) % 500 == 0:
            with open("cached_rating_backup.json", "w") as f:
                f.write(json.dumps(CACHED_RATINGS))

        if len(CACHED_RATINGS) % 100 == 0:
            with open("cached_rating.json", "w") as f:
                f.write(json.dumps(CACHED_RATINGS))

    example["edu_score"] = score

    return example


if __name__ == "__main__":
    fingerweb: datasets.Dataset = gen_from_glot()
    # fingerweb: datasets.Dataset = fingerweb.filter(lambda x: len(x["text"]) < 8192)

    fingerweb = fingerweb.map(map_scorer, num_proc=1)

    fingerweb.push_to_hub("pL-Community/GlotCC-V1-Edu-Scores", private=True)
