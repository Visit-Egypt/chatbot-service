FROM python:3.9
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install loguru

COPY . /code/

ENV PYTHONPATH /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]