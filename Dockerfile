FROM python:3.12

WORKDIR /app
COPY ./requirements.txt ./requirements.txt
COPY ./discord_interface.py ./discord_interface.py
COPY ./modules ./modules
COPY ./prompt ./prompt
COPY *.txt .

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

CMD ["python", "discord_interface.py"]