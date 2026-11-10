FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 HOME=/var/data
RUN mkdir -p /var/data
WORKDIR /app
COPY engine/ /app/
ENTRYPOINT ["python3", "cloud_start.py"]
