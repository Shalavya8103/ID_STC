FROM python:3.8
COPY ./requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install flask
RUN pip install opencv-python
RUN pip install opencv-contrib-python  
RUN pip install -r requirements.txt

COPY ./ app 

EXPOSE 5000

WORKDIR /app

ENTRYPOINT [ "python" ]

CMD [ "./app.py","--host","0.0.0.0","--port","5000" ]
