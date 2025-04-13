from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import random
import smtplib
from email.mime.text import MIMEText
import os
import json
from datetime import datetime, timedelta

app = FastAPI()


USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

# Models
class UserRegister(BaseModel):
    mobile: str
    email: str
    password: str

class UserLogin(BaseModel):
    mobile: str
    password: str

class ForgotPassword(BaseModel):
    mobile_or_email: str


def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def generate_token(mobile):
    return f"token_{mobile}_{datetime.utcnow().timestamp()}"  

def generate_reset_token():
    return f"reset_{random.randint(100000, 999999)}_{datetime.utcnow().timestamp()}"

# APIs
@app.post("/register")
async def register(user: UserRegister):
    users = load_users()
    if user.mobile in users or user.email in [u.get("email") for u in users.values()]:
        raise HTTPException(status_code=400, detail="Mobile or email already registered")
    users[user.mobile] = {"email": user.email, "password": user.password, "reset_token": None, "reset_expiry": None}
    save_users(users)
    return {"message": "User registered successfully", "status": 200}

@app.post("/login")
async def login(user: UserLogin):
    users = load_users()
    user_data = users.get(user.mobile)
    if not user_data or user_data.get("password") != user.password: 
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = generate_token(user.mobile)
    return {"access_token": token, "token_type": "bearer", "status": 200}

@app.post("/forgot_password")
async def forgot_password(data: ForgotPassword):
    users = load_users()
    user_found = False
    reset_token = generate_reset_token()
    reset_expiry = (datetime.now() + timedelta(minutes=15)).isoformat()

    for mobile, user_data in users.items():
        if mobile == data.mobile_or_email or user_data.get("email") == data.mobile_or_email:
            user_found = True
            users[mobile] = {**user_data, "reset_token": reset_token, "reset_expiry": reset_expiry}
            save_users(users)
           
            msg = MIMEText(f"Reset your password: http://localhost/reset?token={reset_token}")
            msg['Subject'] = 'Password Reset'
            msg['From'] = 'your_email@gmail.com'
            msg['To'] = user_data["email"]  
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login("your_email@gmail.com", "your_app_password")
                server.send_message(msg)
            break

    if not user_found:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Reset link sent", "status": 200}

@app.post("/reset_password")
async def reset_password(request: Request):
    data = await request.json()  
    token = data.get("token")
    new_password = data.get("new_password")
    if not token or not new_password:
        raise HTTPException(status_code=400, detail="Token and password required")

    users = load_users()
    for mobile, user_data in users.items():
        if user_data.get("reset_token") == token and datetime.fromisoformat(user_data.get("reset_expiry")) > datetime.now():
            users[mobile] = {**user_data, "password": new_password, "reset_token": None, "reset_expiry": None}
            save_users(users)
            return {"message": "Password reset successful", "status": 200}
    raise HTTPException(status_code=400, detail="Invalid or expired token")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)