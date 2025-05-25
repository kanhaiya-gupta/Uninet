from fastapi import FastAPI, Request, HTTPException, Depends, status, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import uvicorn
import sqlite3
import re

app = FastAPI(
    title="Uninet ML Dashboard",
    description="A unified machine learning framework dashboard",
    version="1.0.0"
)

# Mount templates directory
templates = Jinja2Templates(directory="templates")

# Security configuration
SECRET_KEY = "your-secret-key-here"  # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# Password validation
def validate_password(password: str) -> bool:
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True

# User authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return None
    
    try:
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
    except JWTError:
        return None
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    
    if user is None:
        return None
    return {"id": user[0], "name": user[1], "email": user[2]}

# Authentication routes
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = await get_current_user(request)
    if user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("login.html", {"request": request, "user": None})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()

    if not user or not verify_password(password, user[3]):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "user": None, "error": "Invalid email or password"}
        )

    access_token = create_access_token(
        data={"sub": user[2]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    return response

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    user = await get_current_user(request)
    if user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("signup.html", {"request": request, "user": None})

@app.post("/signup")
async def signup(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    if password != confirm_password:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "user": None, "error": "Passwords do not match"}
        )

    if not validate_password(password):
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "user": None, "error": "Password does not meet requirements"}
        )

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, get_password_hash(password))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "user": None, "error": "Email already registered"}
        )
    conn.close()

    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    return response

# Protected routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user": user}
    )

@app.get("/ml-tasks", response_class=HTMLResponse)
async def ml_tasks(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        "ml_tasks.html",
        {"request": request, "user": user}
    )

@app.get("/ml-tasks/{task_id}", response_class=HTMLResponse)
async def task_detail(request: Request, task_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if task_id not in TASK_DATA:
        return templates.TemplateResponse(
            "404.html",
            {"request": request, "user": user}
        )
    
    return templates.TemplateResponse(
        "task_detail.html",
        {
            "request": request,
            "user": user,
            **TASK_DATA[task_id]
        }
    )

@app.get("/neural-networks", response_class=HTMLResponse)
async def neural_networks(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        "neural_networks.html",
        {"request": request, "user": user}
    )

@app.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        "analytics.html",
        {"request": request, "user": user}
    )

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "user": user}
    )

# Task data for ML tasks
TASK_DATA = {
    "classification": {
        "task_name": "Classification",
        "task_description": "Predicting discrete class labels from input data",
        "overview": "Classification is a supervised learning task where the goal is to predict the category or class of an input based on its features. It's one of the most common machine learning tasks, used in applications ranging from spam detection to medical diagnosis.",
        "use_cases": [
            "Spam Detection: Identifying unwanted emails",
            "Image Recognition: Classifying objects in images",
            "Sentiment Analysis: Determining the sentiment of text",
            "Disease Diagnosis: Predicting medical conditions"
        ],
        "implementation_code": """from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)""",
        "best_practices": [
            "Ensure balanced classes or use appropriate sampling techniques",
            "Perform feature engineering to improve model performance",
            "Use cross-validation to prevent overfitting",
            "Monitor model performance with appropriate metrics (accuracy, precision, recall, F1)"
        ]
    },
    "regression": {
        "task_name": "Regression",
        "task_description": "Predicting continuous values from input data",
        "overview": "Regression is a supervised learning task that predicts continuous numerical values. It's widely used in forecasting, trend analysis, and any scenario where you need to predict a quantity.",
        "use_cases": [
            "House Price Prediction: Estimating property values",
            "Temperature Forecasting: Predicting weather patterns",
            "Stock Price Prediction: Forecasting market trends",
            "Demand Forecasting: Predicting product demand"
        ],
        "implementation_code": """from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)""",
        "best_practices": [
            "Check for linearity between features and target",
            "Handle outliers appropriately",
            "Use regularization to prevent overfitting",
            "Evaluate model performance with RMSE, MAE, and R² metrics"
        ]
    },
    "clustering": {
        "task_name": "Clustering",
        "task_description": "Grouping similar data points together",
        "overview": "Clustering is an unsupervised learning task that groups similar data points together without predefined labels. It's useful for discovering patterns and structures in data.",
        "use_cases": [
            "Customer Segmentation: Grouping customers by behavior",
            "Gene Expression Analysis: Identifying gene patterns",
            "Document Clustering: Organizing similar documents",
            "Image Segmentation: Dividing images into meaningful regions"
        ],
        "implementation_code": """from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and fit model
model = KMeans(n_clusters=3)
clusters = model.fit_predict(X_scaled)""",
        "best_practices": [
            "Preprocess and scale data appropriately",
            "Choose the right number of clusters using elbow method or silhouette analysis",
            "Handle high-dimensional data with dimensionality reduction",
            "Validate clustering results with domain knowledge"
        ]
    },
    "dimensionality-reduction": {
        "task_name": "Dimensionality Reduction",
        "task_description": "Reducing feature space while preserving information",
        "overview": "Dimensionality reduction techniques help reduce the number of features in a dataset while maintaining important information. This is crucial for handling high-dimensional data and improving model performance.",
        "use_cases": [
            "Feature Compression: Reducing data storage requirements",
            "Data Visualization: Visualizing high-dimensional data",
            "Noise Reduction: Removing irrelevant features",
            "Pattern Recognition: Identifying key patterns in data"
        ],
        "implementation_code": """from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)""",
        "best_practices": [
            "Scale data before applying dimensionality reduction",
            "Choose appropriate number of components based on explained variance",
            "Consider both linear (PCA) and non-linear (t-SNE) methods",
            "Validate results by checking reconstruction error"
        ]
    },
    "physics-based-modeling": {
        "task_name": "Physics-Based Modeling",
        "task_description": "Incorporating physical laws into neural networks",
        "overview": "Physics-based modeling combines traditional physics equations with neural networks to create models that respect physical laws while learning from data. This approach is particularly useful in scientific and engineering applications.",
        "use_cases": [
            "Fluid Dynamics: Modeling fluid flow and turbulence",
            "Quantum Mechanics: Predicting quantum states",
            "Climate Modeling: Simulating climate patterns",
            "Material Science: Predicting material properties"
        ],
        "implementation_code": """import torch
import torch.nn as nn

class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# Create model
model = PhysicsInformedNN()""",
        "best_practices": [
            "Incorporate physical constraints in the loss function",
            "Use appropriate activation functions for the physics domain",
            "Balance data-driven and physics-based terms",
            "Validate results against known physical solutions"
        ]
    },
    "graph-based-learning": {
        "task_name": "Graph-Based Learning",
        "task_description": "Learning from graph-structured data",
        "overview": "Graph-based learning deals with data that can be represented as graphs, where nodes represent entities and edges represent relationships. This approach is powerful for analyzing interconnected data.",
        "use_cases": [
            "Social Network Analysis: Understanding social connections",
            "Molecular Structure: Analyzing chemical compounds",
            "Recommendation Systems: Building user-item networks",
            "Knowledge Graphs: Representing and reasoning with knowledge"
        ],
        "implementation_code": """import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, 16)
        self.conv2 = gnn.GCNConv(16, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Create model
model = GNN()""",
        "best_practices": [
            "Choose appropriate graph neural network architecture",
            "Handle different types of graph structures",
            "Consider both node and edge features",
            "Use graph-specific evaluation metrics"
        ]
    }
}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
