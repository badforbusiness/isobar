FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY quantum_solver.py .
CMD ["python", "quantum_solver.py"]
