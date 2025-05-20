# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"

# Worker processes
workers = 6
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout
timeout = 120  # 120 seconds

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"

# Process naming
proc_name = "financial-chatbot"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None 