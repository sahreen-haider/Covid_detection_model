FROM python:3.11-alpine
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN apt-get -y update && apt-get libgl1-mesa-dev 
CMD streamlit viz_app.python
