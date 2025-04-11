import os
import csv
from time import sleep
from typing import List

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
import datetime

load_dotenv()  # take environment variables from .env.


def split_text(text: str) -> List[str]:
    # Split text from the book for every tenth lines

    splitted_text = text.split("\n\n")
    ret = [x.replace('\n', '') for x in splitted_text]

    return ret


def main():
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "ministral-3b-latest"

    client = MistralClient(api_key=api_key)

    system_message = open("prompt/clean_text/system.txt", "r", encoding="utf-8").read()
    user_message = open("prompt/clean_text/user.txt", "r", encoding="utf-8").read()

    text_from_book = open(f"data/bim_eighteenth-century_review-of-the-constituti_oswald-john_1792_djvu.txt", "r",
                          encoding="utf-8").read()
    output_fn = f"data/{str(datetime.datetime.now().timestamp())}_review_of_constitution.csv"
    paragraphs = split_text(text_from_book)

    with open(output_fn, "a", encoding="utf-8") as f:
        fieldnames = ['paragraph_indice', 'orginal_paragraph', "corrected_text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, p in enumerate(paragraphs):
            i = i
            print(f"Process {i+1}/{len(paragraphs)}")
            sleep(1)
            messages = [
                ChatMessage(role="system", content=system_message),
                ChatMessage(role="user", content=user_message.replace("{paragraph}", p))
            ]

            chat_response = client.chat(
                model=model,
                messages=messages,
                temperature=0
            )
            cr = chat_response.choices[0].message.content
            cr = cr.replace("\n", "")
            writer.writerow(
                {"paragraph_indice": i, "orginal_paragraph": p, "corrected_text": cr}
            )
            print(cr)


if __name__ == "__main__":
    main()
