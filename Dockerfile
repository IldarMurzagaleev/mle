FROM python:3.8-slim

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

ENV PYTHONPATH /app/

EXPOSE 5000

CMD [ "bash", "-c"  "bash -c ", "python src/preprocess.py && python src/train.py && python src/predict.py -m LOG_REG -t func && coverage run src/unit_tests/test_preprocess.py && coverage run -a src/unit_tests/test_training.py && coverage report -m"; "gunicorn", "--bind", "0.0.0.0:5000", "src.web_predict:app" ]