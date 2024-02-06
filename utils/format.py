from TOKENS import PROMPTER, BOT, END, SYSTEM
def convert_to_sharegpt(datas: list[str]):
    """
    converting text to sharegpt format
    """
    result = []
    for text in datas:
        conv = []
        elements = text.split(END)
        for element in elements:
            if PROMPTER in element:
                conv.append({"from": "human", "value": element.replace(PROMPTER, "")})
            elif BOT in element:
                conv.append({"from": "gpt", "value": element.replace(BOT, "")})
            elif SYSTEM in element:
                conv.append({"from": "system", "value": element.replace(SYSTEM, "")})
        result.append(conv)
    return result

