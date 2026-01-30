FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5031

# ✅ Chỉ thay dòng CMD
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5031", \
     "--workers", "4", \
     "--worker-class", "gevent", \
     "--worker-connections", "1000", \
     "--timeout", "120", \
     "tool_search_hybrid:app"]