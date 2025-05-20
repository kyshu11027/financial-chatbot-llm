# Finance Chatbot API

A FastAPI-based backend for the finance chatbot application.

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

To start the server, run:

```bash
gunicorn -c gunicorn.conf.py main:app
```

The server will start at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- Swagger UI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`
