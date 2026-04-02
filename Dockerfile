FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download(punkt_tab', quiet=True)"

COPY ..
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

