FROM python:3.9
WORKDIR /code

COPY . /code/
RUN python -m venv env
RUN env/bin/pip install --upgrade pip
RUN env/bin/pip install --no-cache-dir -r requirements.txt
CMD . env/bin/activate && exec uvicorn main:app --host 0.0.0.0 --port 80