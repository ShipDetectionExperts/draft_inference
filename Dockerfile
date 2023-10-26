# Build stage
FROM python:3.10 AS build

WORKDIR /app

COPY package/* /app/

RUN apt-get update && apt-get install -y mesa-utils && \
    pip install -r requirements_prod.txt

# Final stage
FROM python:3.10-alpine

WORKDIR /app

COPY --from=build /app /app

ENV PATH="/app:${PATH}"