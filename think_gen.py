SYS_MSG = """ANALYSIS AND PROBLEM-SOLVING FRAMEWORK

1. TASK CLASSIFICATION
    Determine if the task is:
    a) Logical/Analytical (requires systematic problem-solving)
    b) Creative/Open-ended (requires imagination and originality)

2. FOR LOGICAL/ANALYTICAL TASKS:

    A. PROBLEM DECOMPOSITION
        - State the exact problem in one clear sentence
        - List all given information and constraints
        - Identify any missing information or assumptions needed
        - Define the expected output format

    B. COMPONENT ANALYSIS
        1. Primary Components:
            - List all main elements that directly affect the solution
            - Define their exact properties, units, and ranges
            - Note any constraints or limitations

        2. Secondary Components:
            - List supporting elements or factors
            - Define their influence on primary components
            - Document any dependencies

        3. Implicit Components:
            - Identify hidden assumptions
            - List unstated but necessary conditions
            - Document edge cases to consider

    C. RELATIONSHIP MAPPING
        1. Direct Relationships (1:1):
            - Document exact cause-effect pairs
            - List mathematical relationships
            - Define input-output correlations

        2. Indirect Relationships (1:n):
            - Map cascade effects
            - Document ripple impacts
            - List dependent variables

        3. Hierarchical Relationships:
            - Create parent-child structure
            - Define inheritance patterns
            - Document priority order

        4. Causal Relationships:
            - Map cause-effect chains
            - Document triggers and results
            - Define feedback loops

        5. Temporal Relationships:
            - Document time-dependent factors
            - List sequence requirements
            - Define timing constraints

    D. COMPREHENSION LEVELS
        1. Literal Understanding:
            - Basic facts and given information
            - Explicit statements
            - Direct requirements

        2. Contextual Understanding:
            - Domain-specific knowledge
            - Environmental factors
            - Situational constraints

        3. Abstract Understanding:
            - Pattern recognition
            - Underlying principles
            - General concepts

    E. SOLUTION DEVELOPMENT
        1. Assumptions:
            - List all assumptions explicitly
            - Justify each assumption
            - Document impact on solution

        2. Solution Steps:
            - Number each step
            - Provide detailed sub-steps
            - Include validation points

        3. Boundary Conditions:
            - Define valid input ranges
            - List exception cases
            - Document error handling

    F. SOLUTION VERIFICATION
        1. Logical Consistency:
            - Check step connections
            - Verify mathematical operations
            - Confirm logical flow

        2. Completeness:
            - Check all requirements met
            - Verify all cases handled
            - Confirm no missing steps

        3. Edge Cases:
            - Test minimum values
            - Test maximum values
            - Test boundary conditions

        4. Contradictions:
            - Check for logical conflicts
            - Verify consistent assumptions
            - Confirm no paradoxes

    USE TAGS:
    - Wrap each step explanation in <step> </step>
    - Wrap final solution in <answer> </answer>

3. FOR CREATIVE TASKS:
    - Focus on originality and innovation
    - Provide clear reasoning for creative choices
    - Wrap solution in <answer> </answer>

FEEDBACK SYSTEM:
    - User will provide feedback using <feedback> </feedback>
    - Use feedback to improve subsequent steps
    - Adjust approach based on feedback received

You are only allowed to generated one step at a time. Wait for the feedback before generating the next step.
Each step should be a short as possible. Make many small steps instead of a few large steps. One step should only contain a single idea.
If the task has a time based factor draw the states step by step. This needs to be done for example when solving leaderboards.

Examples:
<step> To find all multiples of 3 that are less than 10, we need to multiply 3 until its larger than 10.</step>
<step> Then we start with 3.</step>
<step> Then we add 3 to 3.</step>
<step> 3 + 3 = 6</step>
<step> Then we add 3 to 6.</step>
<step> 6 + 3 = 9</step>
<step> Then we add 3 to 9.</step>
<step> 9 + 3 = 12</step>
<step> Since 12 is larger than 10, we are done.</step>
<answer>3, 6, 9</answer>
"""

REWARDER = """You are an logical classifier.
The user will provide an question and steps of how to solve the question.
Your task is to critical rate the step of the user.
Answer with <feedback> </feedback> tags.

If the step is correct, write "Correct".
If the step is wrong, write "Wrong".
If the step is more than one fact at a time, write "Too much information in one step. Split this up into smaller parts and think again".

You are only allowed to answer with one of the three options. No more comments are allowed.

Examples:

<step> Um alle vielfachen von 3 die kleiner als 10 sind zu finden, muss man 3, 6 und 9 addieren. </step>
<feedback> Too much information in one step. Split this up into smaller parts and think again </feedback>
<step> Um alle vielfachen von 3 die kleiner als 10 sind zu finden muss man 3 so lange addieren bis man größer als 10 ist. </step>
<feedback> Correct </feedback>
<step> Dann beginnen wir mit 3 </step>
<feedback> Correct </feedback>
<step> Dann addieren wir 3 zu 3 </step>
<feedback> Correct </feedback>
<step> 3 + 3 = 6 </step>
<feedback> Correct </feedback>
<step> Dann addieren wir 3 zu 6 </step>
<feedback> Correct </feedback>
<step> 6 + 3 = 9 </step>
<feedback> Correct </feedback>
<step> Dann addieren wir 3 zu 9 </step>
<feedback> Correct </feedback>
<step> 9 + 3 = 12 </step>
<feedback> Correct </feedback>
<step> Da 12 größer als 10 ist, sind wir fertig. </step>
<feedback> Correct </feedback>
<answer> 3, 6, 9 </answer>

The <step> tags are the steps of the user. The <feedback> tags are the feedback of the rewarder which is your job! The <answer> tags are the answer of the user. Your are not allowed to use the <answer> tags or even the <step> tags. Only the <feedback> tags are allowed.
Dont reply with any other tags than the <feedback> tags and its content.
"""
from dotenv import load_dotenv

load_dotenv()


import hashlib
import json
import os

import datasets
from openai import OpenAI

client = OpenAI()

MODEL_TO_USE = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_TO_USE = "model"

try:
    os.mkdir("logging")
except FileExistsError:
    pass


def think_about(prompt: str):
    msg_hash = hashlib.md5(prompt.encode()).hexdigest()

    # check if we already have a log for this prompt
    if os.path.exists(f"logging/{msg_hash}.json"):
        with open(f"logging/{msg_hash}.json", "r") as f:
            log = json.load(f)
            for step in log["steps"]:
                yield step
        return

    step_list = []
    reward_list = []
    full_answer = ""

    x = 0
    while x < 50:
        messages = [
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": prompt},
        ]

        for i in range(len(step_list)):
            messages.extend(
                [
                    {"role": "assistant", "content": step_list[i]},
                    {"role": "user", "content": reward_list[i]},
                ]
            )

        messages_reward = [
            {"role": "system", "content": REWARDER},
            {"role": "user", "content": prompt},
        ]

        for i in range(len(step_list)):
            messages_reward.extend(
                [
                    {"role": "user", "content": step_list[i]},
                    {"role": "assistant", "content": reward_list[i]},
                ]
            )

        completion = client.chat.completions.create(
            model=MODEL_TO_USE,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        step = completion.choices[0].message.content
        step_list.append(step)

        completion = client.chat.completions.create(
            model=MODEL_TO_USE,
            messages=messages_reward,
            temperature=0.1,
            max_tokens=12,
        )
        reward = completion.choices[0].message.content
        reward_list.append(reward)

        if "<feedback>" not in reward:
            messages_reward.append(
                {
                    "role": "system",
                    "content": "Please provide feedback using <feedback> </feedback> tags.",
                }
            )
            completion = client.chat.completions.create(
                model=MODEL_TO_USE,
                messages=messages_reward,
                temperature=0.1,
                max_tokens=12,
            )
            reward = completion.choices[0].message.content
            reward_list[-1] = reward

        x += 1
        if "Correct" in reward:
            full_answer += step + "\n"
            yield step

        if "<answer>" in step:
            yield step
            with open(f"logging/{msg_hash}.json", "w+") as f:
                f.write(
                    json.dumps(
                        {
                            "steps": step_list,
                            "rewards": reward_list,
                            "prompt": prompt,
                            "full_answer": full_answer,
                        }
                    )
                )
            break


def get_prompt_example(context_input):
    msgs = [
        {
            "role": "system",
            "content": """Du bist ein Fragengenerator für Universitätsprofessoren.
Deine Aufgabe ist es einen Text vom Nutzer zu nehmen und sehr komplexe, nicht offensichtlich zu beantwortende Fragen zu erstellen.
Die Antwort für die von dir erstellten Fragen muss aus mehreren Informationen des Textes vom Nutzer abgeleitet werden.
Der Grad der Komplexität sollte Universitätslevel haben und mehrschrittige Argumentationen beinhalten.
Jede Frage muss mit einer eindeutig nachprüfbaren Antwort beantwortet werden können.
Du darfst nur mit der Frage antworten und keine weiteren Informationen geben.

Beispiele für die komplexität der Fragen:
- Welche Partei hat die längste Zeit den Kanzler gestellt und wieviele Jahren waren das ?
- Wenn ich 3 Kerzen von selber Länge gleichzeitig anzünde und am Ende 3 unterschiedliche Längen habe, welche habe ich zuerst ausgepustet ? Kerze 1 hat 5cm Länge, Kerze 2 hat 7cm Länge und Kerze 3 hat 2cm Länge.
- Klassifiziere die gegebenen Elemente in verschiedene Kategorien und erkläre die Gründe für die Zuordnung. Wähle aus den folgenden Kateogrien: [Kategorie 1], [Kategorie 2], [Kategorie 3].
- Zeige die Zusammenhänge zwischen allen Personen auf und erkläre in welchem Verhältnis sie zueinander stehen. Liste alle Beziehungen für jede Person schritt für Schritt auf.
- Welche Sicherheiten kann die Bank bei einem Kredit verlangen und wie wirkt sich dies auf die Kreditwürdigkeit aus wenn diese jeweils wegfallen? Liste die Sicherheiten auf und erkläre deren Auswirkungen auf die Kreditwürdigkeit.
- Welche technischen Herausforderungen und Nachteile hatte die Heinkel He 119, die sie letztendlich für eine militärische Verwendung ungeeignet machten, und wie versuchte Heinkel, diese Probleme zu überwinden oder das Flugzeug für andere Zwecke zu adaptieren? Welche Funktionen hätte man sich von den Konkurrenzmodellen übernehmen können, um die He 119 zu verbessern?""",
        },
        {
            "role": "user",
            "content": context_input,
        },
    ]

    response = client.chat.completions.create(
        messages=msgs,
        model=MODEL_TO_USE,
        temperature=0.1,
    )

    result = response.choices[0].message.content
    return result


ds = datasets.load_dataset(
    "flozi00/Fineweb2-German-Eduscore-4andMore", split="train", streaming=True
).filter(lambda x: len(x["text"]) < 10_000)

for example in ds:
    question = get_prompt_example(example["text"])
    print(question)
    for step in think_about(example["text"] + "\n\n" + question):
        print(step)

    print("\n\n")
