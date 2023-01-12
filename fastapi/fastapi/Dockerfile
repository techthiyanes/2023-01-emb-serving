FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

RUN pip install torch  --extra-index-url https://download.pytorch.org/whl/cu117
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./src /app