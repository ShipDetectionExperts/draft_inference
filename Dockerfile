FROM python:3.10

WORKDIR /app

COPY package/* /app

RUN apt-get update && apt-get install -y mesa-utils

RUN pip install -r requirements_prod.txt

ENV PATH="/app:${PATH}"