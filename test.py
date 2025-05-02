import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# List of all environment variables used in your app
env_vars = [
    "OPENAI_API_KEY",
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_API_KEY",
    "KAFKA_API_SECRET",
    "MONGODB_URI"
]

# Print each variable and whether it's loaded correctly
print("🔍 Environment Variable Check:\n")
for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"{var} = {value}")
    else:
        print(f"{var} = ❌ NOT FOUND")

# Optional: warn if URI has issues
mongo_uri = os.getenv("MONGO_URI")
if mongo_uri and mongo_uri.strip() == "":
    print("\n⚠️ Warning: MONGO_URI is an empty string.")
