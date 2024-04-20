FROM python:3.11-slim
WORKDIR $HOME/app
COPY requirements.txt $HOME/app
RUN mkdir /.cache && chmod 777 /.cache
RUN pip install -r requirements.txt
COPY . $HOME/app
EXPOSE 23333
CMD ["python", "-m", "apis.chat_api"]