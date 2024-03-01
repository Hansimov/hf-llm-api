FROM python:3.11-slim
WORKDIR $HOME/app
COPY . .
RUN mkdir /.cache && chmod 777 /.cache
RUN pip install -r requirements.txt
VOLUME /data
EXPOSE 23333
CMD ["python", "-m", "apis.chat_api"]