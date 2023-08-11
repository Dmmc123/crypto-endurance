FROM python:3.11.4-slim

RUN mkdir /app

COPY . /app

WORKDIR /app

COPY poetry.lock pyproject.toml /app

RUN pip install poetry

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

CMD streamlit run ts/gui/dashboard.py