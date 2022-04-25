FROM python:3.8-slim

RUN mkdir -p /app

COPY .  /app/
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1
RUN pip install -r requirements.txt
EXPOSE 8000
CMD [ "main.py" ]
ENTRYPOINT [ "python" ]
