# 1. Use a lightweight Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the dependency file first (for caching speed)
COPY requirements.txt .

# 4. Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files into the container
COPY . .

# 6. Expose Port 5000 (The port Flask uses)
EXPOSE 5000

# 7. Define the command to run your app
CMD ["python", "app.py"]