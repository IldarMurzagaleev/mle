FROM python:3.8-slim

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

ENV PYTHONPATH /app/

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "src.web_predict:app" ]