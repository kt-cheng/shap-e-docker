FROM python:3.11

WORKDIR /app

COPY ./shap-e/ /app/
COPY ./example/ /app/

RUN apt-get update && apt-get install -y python3 python3-pip vim && pip3 install --upgrade pip
RUN pip install -e .
RUN pip install pyyaml ipywidgets

CMD tail -f /dev/null