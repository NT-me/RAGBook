FROM python:3.12

WORKDIR /app
COPY ./requirements.txt ./requirements.txt
COPY ./discord_interface.py ./discord_interface.py
COPY ./prompt ./prompt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

CMD ["python", "discord_interface.py"]