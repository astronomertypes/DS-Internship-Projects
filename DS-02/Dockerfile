# Small Python base image
FROM python:3.11-slim

# Work inside /app in the container
WORKDIR /app

# Only the libraries the dashboard needs (no TensorFlow/PyTorch/Prophet)
RUN pip install --no-cache-dir dash plotly pandas pyarrow gunicorn

# Copy the dashboard code and its data (app/app.py and app/data/)
COPY app/ /app/

# The dashboard listens on port 8050
EXPOSE 8050

# gunicorn is a production web server; app:server = file app.py, variable "server"
CMD ["gunicorn", "-b", "0.0.0.0:8050", "app:server"]