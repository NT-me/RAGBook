import os
import csv
from time import sleep
from typing import List

from mistralai.client import MistralClient
from itertools import batched
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


def split_text(text: str) -> List[str]:
    # Split text from the book for every tenth lines

    splitted_text = text.split("\n")
    unflattened = list(batched(splitted_text, 10))
    ret = [" ".join(x) for x in unflattened]

    return ret


def main():
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "open-mistral-7b"

    client = MistralClient(api_key=api_key)

    system_message = open("prompt/clean_text/system.txt", "r", encoding="utf-8").read()

    text_from_book = open(f"data/bim_eighteenth-century_the-cry-of-nature-or-a_oswald-john-miscellane_1791_djvu.txt", "r",
                          encoding="utf-8").read()
    start_indice = 445
    paragraphs = split_text(text_from_book)[start_indice:]

    with open("data/output2.csv", "a", encoding="utf-8") as f:
        fieldnames = ['orginal_indice', 'orginal_batch10', "corrected_text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        #writer.writeheader()

        for i, p in enumerate(paragraphs):
            i = i +start_indice
            print(f"Process {i+1}/{len(paragraphs)+start_indice}")
            sleep(1)
            messages = [
                ChatMessage(role="system", content=system_message),
                ChatMessage(role="user", content=p)
            ]

            chat_response = client.chat(
                model=model,
                messages=messages,
                temperature=0
            )
            cr = chat_response.choices[0].message.content
            cr = cr.replace("\n", "")
            writer.writerow(
                {"orginal_indice": i, "orginal_batch10": p, "corrected_text": cr}
            )


if __name__ == "__main__":
    main()
