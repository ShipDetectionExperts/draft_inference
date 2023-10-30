FROM python:3.10 AS build

WORKDIR /app

COPY package/ /app/

RUN pip install --no-cache-dir -r requirements_prod.txt

FROM python:3.10

WORKDIR /app

COPY --from=build /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

RUN apt-get update && apt-get install -y mesa-utils

COPY --from=build /app/ /app/

ENV PATH="/app:${PATH}"