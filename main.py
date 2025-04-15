from email.mime.text import MIMEText
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
import random
from datetime import datetime, timedelta
import os
import shutil
import uuid
import smtplib
from typing import Optional, List
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import for traffic accident predictor
import tensorflow as tf
import numpy as np
from traffic_predictor import TrafficAccidentPredictor

# Initialize the FastAPI app
app = FastAPI()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Session middleware setup
app.add_middleware(
    SessionMiddleware,
    secret_key="your-secret-key-123",
    session_cookie="session_cookie"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize the predictor
predictor = None
try:
    predictor = TrafficAccidentPredictor()
    predictor.load_model()
    logger.info("Traffic accident prediction model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load traffic accident prediction model: {str(e)}")

# Mount the uploads directory to serve files
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory="templates")

# User database setup
USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

# Email configuration - Set these with your SMTP credentials
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "your_email@gmail.com")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "your_email_password")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "no-reply@yourcomplaintapp.com")

class LocationData(BaseModel):
    lat: float
    lng: float

def send_email_notification(to_email, subject, message):
    """Send email notification using SMTP"""
    try:
        # Create message
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        
        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {str(e)}")
        return False

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def get_current_user(request: Request) -> Optional[str]:
    """Get the current user's mobile number from the session token"""
    token = request.session.get("access_token")
    return get_user_from_token(token) if token else None

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def load_reports():
    if not os.path.exists("reports.json"):
        with open("reports.json", "w") as f:
            json.dump({}, f)
    with open("reports.json", "r") as f:
        return json.load(f)

def save_reports(reports):
    with open("reports.json", "w") as f:
        json.dump(reports, f)

def generate_token(mobile):
    return f"token_{mobile}_{datetime.utcnow().timestamp()}"

# Helper function to get user mobile from session token
def get_user_from_token(token):
    if not token:
        return None
    parts = token.split('_')
    if len(parts) >= 2:
        return parts[1]
    return None

# Helper function for saving uploaded files
def save_upload_file(upload_file: UploadFile) -> str:
    # Generate a unique filename
    file_extension = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".jpg"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    # Return the relative path to access the file
    return f"/uploads/{unique_filename}"

# Authentication endpoints
@app.post("/register")
async def register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    mobile: str = Form(...),
    password: str = Form(...)
):
    # Password validation
    if len(password) < 8:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Password must be at least 8 characters"},
            status_code=400
        )
    
    users = load_users()
    if mobile in users or email in [u.get("email") for u in users.values()]:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Mobile or email already registered"},
            status_code=400
        )

    users[mobile] = {
        "name": name,
        "email": email,
        "password": password,
        "reset_token": None,
        "reset_expiry": None
    }
    save_users(users)
    
    response = RedirectResponse(url="/home", status_code=303)
    request.session["access_token"] = generate_token(mobile)
    return response

@app.post("/login")
async def login(
    request: Request,
    mobile: str = Form(...),
    password: str = Form(...)
):
    users = load_users()
    user_data = users.get(mobile)
    
    if not user_data or user_data.get("password") != password:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid mobile or password"},
            status_code=401
        )

    response = RedirectResponse(url="/home", status_code=303)
    request.session["access_token"] = generate_token(mobile)
    return response

@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)

# Other endpoints
@app.get("/")
async def root():
    return RedirectResponse(url="/home")

@app.get("/home")
async def home(request: Request):
    if not request.session.get("access_token"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/login")
async def login_page(request: Request, error: str = None):
    if request.session.get("access_token"):
        return RedirectResponse(url="/home")
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@app.get("/signup")
async def signup_page(request: Request, error: str = None):
    if request.session.get("access_token"):
        return RedirectResponse(url="/home")
    return templates.TemplateResponse("signup.html", {"request": request, "error": error})

@app.get("/accident-risk-map")
async def accident_risk_map(request: Request):
    """Render the accident risk map page"""
    # Load all reports to display on the map
    reports = load_reports()
    return templates.TemplateResponse(
        "accidentrisk.html", 
        {"request": request, "reports": reports}
    )

@app.post("/predict-accident")
async def predict_accident(location: LocationData):
    """Predict accident risk for a given location"""
    if predictor is None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Prediction model not available"}
        )
    
    try:
        # Convert to the format expected by the predictor
        location_dict = {"lat": location.lat, "lng": location.lng}
        
        # Get prediction
        prediction = predictor.predict_accident(location_dict)
        
        if prediction is None:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Failed to get prediction"}
            )
        
        return JSONResponse(
            content={
                "status": "success",
                "prediction": prediction
            }
        )
    except Exception as e:
        logger.error(f"Error during accident prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Prediction error: {str(e)}"}
        )

@app.post("/report")
async def submit_report(request: Request):
    # Check if user is logged in
    access_token = request.session.get("access_token")
    if not access_token:
        return JSONResponse(
            status_code=401,
            content={"message": "Please login to submit a report", "status": "error"}
        )
    
    form_data = await request.form()
    reports = load_reports()
    
    # Generate a unique report ID
    report_id = str(len(reports) + 1)
    
    # Get user mobile from token
    user_mobile = get_user_from_token(access_token)
    
    # Create timestamp
    current_time = datetime.now().isoformat()
    
    # Process photo if present
    photo_path = None
    if "photo" in form_data and form_data["photo"].filename:
        # Save the uploaded file
        photo_file = form_data["photo"]
        photo_path = save_upload_file(photo_file)
    
    # Create the report entry with votes initialized
    reports[report_id] = {
        "id": report_id,
        "user_mobile": user_mobile,
        "issue_type": form_data.get("issueType"),
        "location": form_data.get("location"),
        "latitude": float(form_data.get("latitude", 0)),
        "longitude": float(form_data.get("longitude", 0)),
        "description": form_data.get("description"),
        "photo_path": photo_path,
        "status": "pending",
        "created_at": current_time,
        "updated_at": current_time,
        "votes": {"upvotes": {}, "downvotes": {}},
        "upvote_count": 0,
        "downvote_count": 0,
        "vote_score": 0,
        "resolution_details": ""
    }
    
    save_reports(reports)
    return JSONResponse(content={"status": "success", "report_id": report_id})

@app.get("/my-complaints")
async def user_complaints(request: Request):
    # Check if user is logged in
    access_token = request.session.get("access_token")
    if not access_token:
        return RedirectResponse(url="/login")
    
    # Get user mobile from token
    user_mobile = get_user_from_token(access_token)
    
    # Load all reports
    reports = load_reports()
    
    # Filter reports for this user
    user_reports = {
        report_id: report_data 
        for report_id, report_data in reports.items() 
        if report_data.get("user_mobile") == user_mobile
    }
    
    return templates.TemplateResponse(
        "my-complaints.html", 
        {"request": request, "reports": user_reports}
    )

@app.post("/vote/{report_id}/{vote_type}")
async def vote_report(request: Request, report_id: str, vote_type: str):
    """Handle upvote or downvote for a report"""
    # Check if user is logged in
    current_user = get_current_user(request)
    if not current_user:
        return JSONResponse(
            status_code=401,
            content={"message": "Please login to vote", "status": "error"}
        )
    
    # Validate vote type
    if vote_type not in ["upvote", "downvote"]:
        return JSONResponse(
            status_code=400,
            content={"message": "Invalid vote type", "status": "error"}
        )
    
    # Load reports
    reports = load_reports()
    
    # Check if report exists
    if report_id not in reports:
        return JSONResponse(
            status_code=404,
            content={"message": "Report not found", "status": "error"}
        )
    
    # Initialize votes structure if it doesn't exist
    if "votes" not in reports[report_id]:
        reports[report_id]["votes"] = {"upvotes": {}, "downvotes": {}}
    
    # Handle the vote
    upvotes = reports[report_id]["votes"]["upvotes"]
    downvotes = reports[report_id]["votes"]["downvotes"]
    
    # Check if user already voted
    already_upvoted = current_user in upvotes
    already_downvoted = current_user in downvotes
    
    # Handle different voting scenarios
    if vote_type == "upvote":
        if already_upvoted:
            # Remove upvote (toggle off)
            del upvotes[current_user]
            message = "Upvote removed"
        else:
            # Add upvote and remove downvote if exists
            upvotes[current_user] = datetime.now().isoformat()
            if already_downvoted:
                del downvotes[current_user]
            message = "Upvoted successfully"
    else:  # downvote
        if already_downvoted:
            # Remove downvote (toggle off)
            del downvotes[current_user]
            message = "Downvote removed"
        else:
            # Add downvote and remove upvote if exists
            downvotes[current_user] = datetime.now().isoformat()
            if already_upvoted:
                del upvotes[current_user]
            message = "Downvoted successfully"
    
    # Calculate vote counts
    upvote_count = len(upvotes)
    downvote_count = len(downvotes)
    
    # Update report with vote counts for easy access
    reports[report_id]["upvote_count"] = upvote_count
    reports[report_id]["downvote_count"] = downvote_count
    reports[report_id]["vote_score"] = upvote_count - downvote_count
    reports[report_id]["updated_at"] = datetime.now().isoformat()
    
    # Save reports
    save_reports(reports)
    
    # Return vote counts
    return {
        "status": "success",
        "message": message,
        "upvote_count": upvote_count,
        "downvote_count": downvote_count,
        "vote_score": upvote_count - downvote_count,
        "user_vote": "upvote" if current_user in upvotes else "downvote" if current_user in downvotes else None
    }

@app.get("/report/{report_id}")
async def get_report(request: Request, report_id: str):
    reports = load_reports()
    
    if report_id not in reports:
        return JSONResponse(
            status_code=404,
            content={"message": "Report not found", "status": "error"}
        )
    
    report = reports[report_id]
    
    # Check if current user has voted on this report
    current_user = get_current_user(request)
    user_vote = None
    
    if current_user and "votes" in report:
        if current_user in report["votes"].get("upvotes", {}):
            user_vote = "upvote"
        elif current_user in report["votes"].get("downvotes", {}):
            user_vote = "downvote"
    
    # Add user_vote to response
    response_report = dict(report)
    response_report["user_vote"] = user_vote
    
    return JSONResponse(content={"status": "success", "report": response_report})

# New endpoint to mark a report as resolved
@app.post("/report/{report_id}/resolve")
async def resolve_report(
    request: Request, 
    report_id: str,
    resolution_details: str = Form(...)
):
    # Check if user is logged in and has admin privileges
    # You might want to add admin check here
    current_user = get_current_user(request)
    if not current_user:
        return JSONResponse(
            status_code=401,
            content={"message": "Please login to resolve reports", "status": "error"}
        )
    
    # Load reports
    reports = load_reports()
    
    # Check if report exists
    if report_id not in reports:
        return JSONResponse(
            status_code=404,
            content={"message": "Report not found", "status": "error"}
        )
    
    # Get the report and user info
    report = reports[report_id]
    user_mobile = report["user_mobile"]
    
    # Update report status
    reports[report_id]["status"] = "resolved"
    reports[report_id]["resolution_details"] = resolution_details
    reports[report_id]["resolved_at"] = datetime.now().isoformat()
    reports[report_id]["resolved_by"] = current_user
    reports[report_id]["updated_at"] = datetime.now().isoformat()
    
    # Save reports
    save_reports(reports)
    
    # Get user email from their mobile number
    users = load_users()
    user_data = users.get(user_mobile, {})
    user_email = user_data.get("email")
    user_name = user_data.get("name", "User")
    
    if user_email:
        # Send email notification to the user
        issue_type = report["issue_type"]
        location = report["location"]
        
        subject = f"Report Resolved: {issue_type} at {location}"
        
        message = f"""
Dear {user_name},

Great news! Your report (ID: {report_id}) has been resolved.

Details:
- Issue Type: {issue_type}
- Location: {location}
- Resolution: {resolution_details}
- Resolved on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Thank you for helping improve our community!

Best regards,
The Complaint Portal Team
"""
        
        notification_sent = send_email_notification(user_email, subject, message)
    else:
        notification_sent = False
    
    return JSONResponse(
        content={
            "status": "success", 
            "message": "Report marked as resolved",
            "notification_sent": notification_sent
        }
    )

@app.post("/reset_password")
async def reset_password(request: Request):
    data = await request.json()  
    token = data.get("token")
    new_password = data.get("new_password")
    
    # Password validation
    if len(new_password) < 8:
        raise HTTPException(
            status_code=400,
            detail="New password must be at least 8 characters long"
        )
    if not any(char.isdigit() for char in new_password):
        raise HTTPException(
            status_code=400,
            detail="New password must contain at least one number"
        )
    if not any(not char.isalnum() for char in new_password):
        raise HTTPException(
            status_code=400,
            detail="New password must contain at least one special character"
        )

    users = load_users()
    
    # Confirm password reset
    for mobile, user_data in users.items():
        if user_data.get("reset_token") == token and datetime.fromisoformat(user_data.get("reset_expiry")) > datetime.now():
            users[mobile] = {**user_data, "password": new_password, "reset_token": None, "reset_expiry": None}
            save_users(users)
            return {"message": "Password reset successful", "status": 200}
    return {"message": "Password reset successful", "status": 200}

@app.get('/aboutus')
async def read_root(request: Request):
    return templates.TemplateResponse("aboutus.html", {"request": request})

@app.get('/trackstatus')
async def read_root(request: Request):
    return templates.TemplateResponse("trackstatus.html", {"request": request})

@app.post('/trackstatus')
async def trackstatus_lookup(request: Request, report_id: str = Form(...)):
    # Load reports
    reports = load_reports()
    
    # Check if report exists
    if report_id not in reports:
        return templates.TemplateResponse(
            "trackstatus.html", 
            {"request": request, "error": "Report not found", "report_id": report_id}
        )
    
    # If report exists, pass it to the template
    report = reports[report_id]
    return templates.TemplateResponse(
        "trackstatus.html", 
        {"request": request, "report": report, "report_id": report_id}
    )

@app.get('/contactus')
async def read_root(request: Request):
    return templates.TemplateResponse("contactus.html", {"request": request})

@app.get('/faq')
async def read_root(request: Request):
    return templates.TemplateResponse("FAQs.html", {"request": request})

@app.get('/streetview')
async def streetview(request: Request):
    # Load all reports to display on the map 
    reports = load_reports()
    return templates.TemplateResponse("streetview.html", {"request": request, "reports": reports})

@app.get('/verifypage')
async def read_page(request: Request):
    return templates.TemplateResponse("verifypage.html", {"request": request})

# Admin panel to view and resolve reports
@app.get('/admin/reports')
async def admin_reports(request: Request):
    # Check if admin is logged in (you may want to add actual admin auth)
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/login")
    
    # Load all reports
    reports = load_reports()
    
    return templates.TemplateResponse(
        "admin-reports.html", 
        {"request": request, "reports": reports}
    )

@app.get('/datastats')
async def datastats(request: Request):
    return templates.TemplateResponse('datastatistic.html', {"request" : request})

@app.get('/guides')
async def read_root(request: Request):
    return templates.TemplateResponse('guides.html', {"request" : request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)