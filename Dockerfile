FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends     libblas3     libopenblas0     liblapack3     gfortran     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
