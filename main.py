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

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Neural Network Data
NEURAL_NETWORK_DATA = {
    "fnn": {
        "name": "Feedforward Neural Network (FNN)",
        "description": "Basic dense network for structured data",
        "overview": "Feedforward Neural Networks are the most basic type of neural network architecture. They consist of an input layer, one or more hidden layers, and an output layer. Each layer is fully connected to the next layer, and information flows in one direction - from input to output.",
        "use_cases": [
            "Structured Data Analysis",
            "Tabular Data Processing",
            "Basic Pattern Recognition"
        ],
        "architecture": "Input Layer → Hidden Layers → Output Layer",
        "implementation_code": """import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

# Create model
model = FNN(input_size=10, hidden_size=64, output_size=2)"""
    },
    "cnn": {
        "name": "Convolutional Neural Network (CNN)",
        "description": "Specialized for spatial data processing",
        "overview": "Convolutional Neural Networks are designed to process data with a grid-like topology, such as images. They use convolutional layers to automatically learn spatial hierarchies of features, making them particularly effective for image recognition and computer vision tasks.",
        "use_cases": [
            "Image Classification",
            "Object Detection",
            "Image Segmentation",
            "Video Analysis"
        ],
        "architecture": "Convolutional Layers → Pooling → Fully Connected",
        "implementation_code": """import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# Create model
model = CNN()"""
    },
    "rnn": {
        "name": "Recurrent Neural Network (RNN)",
        "description": "Designed for sequential data processing",
        "overview": "Recurrent Neural Networks are designed to process sequences of data. They maintain an internal state (memory) that captures information about the sequence seen so far, making them suitable for tasks involving time series or natural language processing.",
        "use_cases": [
            "Natural Language Processing",
            "Time Series Analysis",
            "Speech Recognition",
            "Music Generation"
        ],
        "architecture": "LSTM/GRU Cells → Recurrent Connections",
        "implementation_code": """import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Create model
model = RNN(input_size=10, hidden_size=64, num_layers=2, output_size=2)"""
    },
    "transformer": {
        "name": "Transformer",
        "description": "Attention-based architecture for sequence processing",
        "overview": "Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequences of data. They have revolutionized natural language processing and are now being applied to various other domains.",
        "use_cases": [
            "Machine Translation",
            "Text Generation",
            "Question Answering",
            "Image Recognition (ViT)"
        ],
        "architecture": "Self-Attention → Feed Forward → Layer Norm",
        "implementation_code": """import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 10)
    
    def forward(self, src):
        output = self.transformer_encoder(src)
        return self.fc(output)

# Create model
model = Transformer(d_model=512, nhead=8, num_layers=6)"""
    },
    "autoencoder": {
        "name": "Autoencoder",
        "description": "Learns compressed representations of data",
        "overview": "Autoencoders are neural networks designed to learn efficient representations of data through an unsupervised learning process. They consist of an encoder that compresses the input and a decoder that reconstructs the input from the compressed representation.",
        "use_cases": [
            "Data Compression",
            "Feature Learning",
            "Noise Reduction",
            "Anomaly Detection"
        ],
        "architecture": "Encoder → Latent Space → Decoder",
        "implementation_code": """import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

# Create model
model = Autoencoder(input_size=784, hidden_size=128)"""
    },
    "vae": {
        "name": "Variational Autoencoder (VAE)",
        "description": "Probabilistic generative model",
        "overview": "Variational Autoencoders are a type of generative model that learns to encode data into a probability distribution rather than a single point. This allows them to generate new data samples by sampling from the learned distribution.",
        "use_cases": [
            "Image Generation",
            "Data Augmentation",
            "Feature Learning",
            "Unsupervised Learning"
        ],
        "architecture": "Encoder → Latent Distribution → Decoder",
        "implementation_code": """import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_var = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Create model
model = VAE(input_size=784, hidden_size=256, latent_size=32)"""
    },
    "gan": {
        "name": "Generative Adversarial Network (GAN)",
        "description": "Competing networks for data generation",
        "overview": "GANs consist of two neural networks: a generator that creates synthetic data and a discriminator that tries to distinguish between real and synthetic data. Through adversarial training, the generator learns to create increasingly realistic data.",
        "use_cases": [
            "Image Generation",
            "Style Transfer",
            "Data Augmentation",
            "Super Resolution"
        ],
        "architecture": "Generator → Discriminator → Adversarial Training",
        "implementation_code": """import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Create models
generator = Generator(latent_size=100)
discriminator = Discriminator()"""
    },
    "rbfn": {
        "name": "Radial Basis Function Network (RBFN)",
        "description": "Uses radial basis functions for pattern recognition",
        "overview": "RBFNs are a type of neural network that uses radial basis functions as activation functions. They are particularly effective for function approximation and pattern recognition tasks.",
        "use_cases": [
            "Function Approximation",
            "Pattern Recognition",
            "Time Series Prediction",
            "Control Systems"
        ],
        "architecture": "Input → RBF Layer → Output",
        "implementation_code": """import torch
import torch.nn as nn

class RBFN(nn.Module):
    def __init__(self, input_size, num_centers, output_size):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_size))
        self.beta = nn.Parameter(torch.ones(num_centers))
        self.linear = nn.Linear(num_centers, output_size)
    
    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        rbf = torch.exp(-self.beta * distances ** 2)
        return self.linear(rbf)

# Create model
model = RBFN(input_size=10, num_centers=20, output_size=2)"""
    },
    "som": {
        "name": "Self-Organizing Maps (SOM)",
        "description": "Unsupervised learning for dimensionality reduction",
        "overview": "SOMs are a type of neural network that uses competitive learning to create a low-dimensional representation of high-dimensional data. They are particularly useful for data visualization and clustering.",
        "use_cases": [
            "Data Visualization",
            "Pattern Recognition",
            "Document Organization",
            "Image Processing"
        ],
        "architecture": "Competitive Learning → Neighborhood Preservation",
        "implementation_code": """import numpy as np
from minisom import MiniSom

class SOM:
    def __init__(self, input_size, map_size):
        self.som = MiniSom(map_size[0], map_size[1], input_size)
    
    def train(self, data, num_iterations):
        self.som.train_random(data, num_iterations)
    
    def get_winner(self, x):
        return self.som.winner(x)

# Create model
model = SOM(input_size=10, map_size=(10, 10))"""
    },
    "dbn": {
        "name": "Deep Belief Networks (DBN)",
        "description": "Layer-wise pre-trained networks",
        "overview": "DBNs are a type of neural network that consists of multiple layers of Restricted Boltzmann Machines (RBMs). They are trained in a greedy, layer-wise manner and can be fine-tuned using backpropagation.",
        "use_cases": [
            "Feature Learning",
            "Pre-training",
            "Classification",
            "Dimensionality Reduction"
        ],
        "architecture": "RBM Layers → Fine-tuning",
        "implementation_code": """import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, visible_size, hidden_size):
        super().__init__()
        self.W = nn.Parameter(torch.randn(visible_size, hidden_size))
        self.visible_bias = nn.Parameter(torch.zeros(visible_size))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, v):
        h = torch.sigmoid(torch.matmul(v, self.W) + self.hidden_bias)
        v_recon = torch.sigmoid(torch.matmul(h, self.W.t()) + self.visible_bias)
        return v_recon

class DBN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.rbms = nn.ModuleList([
            RBM(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes)-1)
        ])
    
    def forward(self, x):
        for rbm in self.rbms:
            x = rbm(x)
        return x

# Create model
model = DBN([784, 500, 200, 10])"""
    },
    "pinn": {
        "name": "Physics-Informed Neural Network (PINN)",
        "description": "Embeds physical laws into neural networks",
        "overview": "PINNs are neural networks that incorporate physical laws and constraints into their training process. They are particularly useful for solving differential equations and modeling physical systems.",
        "use_cases": [
            "Fluid Dynamics",
            "Quantum Mechanics",
            "Climate Modeling",
            "Material Science"
        ],
        "architecture": "Neural Network + Physical Constraints",
        "implementation_code": """import torch
import torch.nn as nn

class PINN(nn.Module):
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
    
    def compute_derivatives(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        inputs.requires_grad_(True)
        u = self.net(inputs)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u))[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u))[0]
        return u, u_t, u_x

# Create model
model = PINN()"""
    },
    "neural_ode": {
        "name": "Neural ODEs",
        "description": "Models continuous dynamics using neural networks",
        "overview": "Neural ODEs are a type of neural network that models continuous-time dynamics using ordinary differential equations. They are particularly useful for modeling time series and dynamical systems.",
        "use_cases": [
            "Time Series Prediction",
            "Dynamical Systems",
            "Continuous-time Models",
            "Irregular Time Series"
        ],
        "architecture": "Neural Network + ODE Solver",
        "implementation_code": """import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )
    
    def forward(self, t, y):
        return self.net(y)

class NeuralODE(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, y0, t):
        solution = odeint(self.func, y0, t)
        return solution

# Create model
func = ODEFunc()
model = NeuralODE(func)"""
    },
    "gnn": {
        "name": "Graph Neural Network (GNN)",
        "description": "Operates on graph-structured data",
        "overview": "GNNs are neural networks designed to operate on graph-structured data. They use message passing to aggregate information from neighboring nodes and update node representations.",
        "use_cases": [
            "Social Network Analysis",
            "Molecular Structure",
            "Recommendation Systems",
            "Knowledge Graphs"
        ],
        "architecture": "Graph Convolution → Message Passing",
        "implementation_code": """import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Create model
model = GNN(in_channels=10, hidden_channels=64, out_channels=2)"""
    },
    "snn": {
        "name": "Spiking Neural Network (SNN)",
        "description": "Inspired by biological neurons",
        "overview": "SNNs are neural networks that use discrete spikes to transmit information, similar to biological neurons. They are particularly energy-efficient and suitable for neuromorphic computing.",
        "use_cases": [
            "Energy-efficient Computing",
            "Real-time Systems",
            "Neuromorphic Hardware",
            "Event-based Processing"
        ],
        "architecture": "Spiking Neurons → Temporal Dynamics",
        "implementation_code": """import torch
import torch.nn as nn
import snntorch as snn

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=0.9)
    
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        mem2_rec = []
        
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Create model
model = SNN(input_size=10, hidden_size=64, output_size=2)"""
    }
}

@app.get("/neural-networks/{network_id}", response_class=HTMLResponse)
async def neural_network_detail(request: Request, network_id: str):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    if network_id not in NEURAL_NETWORK_DATA:
        return templates.TemplateResponse(
            "404.html",
            {"request": request, "user": user}
        )
    
    # Define available tasks for each network type
    network_tasks = {
        "fnn": [
            {"id": "fnn_classification", "name": "Classification"},
            {"id": "fnn_regression", "name": "Regression"}
        ],
        "cnn": [
            {"id": "cnn_classification", "name": "Image Classification"},
            {"id": "cnn_segmentation", "name": "Image Segmentation"},
            {"id": "cnn_detection", "name": "Object Detection"}
        ],
        "rnn": [
            {"id": "rnn_sequence", "name": "Sequence Modeling"},
            {"id": "rnn_timeseries", "name": "Time Series Prediction"},
            {"id": "rnn_nlp", "name": "Natural Language Processing"}
        ],
        "transformer": [
            {"id": "transformer_nlp", "name": "Natural Language Processing"},
            {"id": "transformer_seq2seq", "name": "Sequence-to-Sequence"},
            {"id": "transformer_timeseries", "name": "Time Series"}
        ],
        "autoencoder": [
            {"id": "autoencoder_compression", "name": "Data Compression"},
            {"id": "autoencoder_anomaly", "name": "Anomaly Detection"}
        ],
        "vae": [
            {"id": "vae_generation", "name": "Data Generation"},
            {"id": "vae_feature", "name": "Feature Learning"}
        ],
        "gan": [
            {"id": "gan_generation", "name": "Data Generation"},
            {"id": "gan_style", "name": "Style Transfer"}
        ],
        "rbfn": [
            {"id": "rbfn_classification", "name": "Classification"},
            {"id": "rbfn_regression", "name": "Regression"}
        ],
        "som": [
            {"id": "som_clustering", "name": "Clustering"},
            {"id": "som_visualization", "name": "Data Visualization"}
        ],
        "dbn": [
            {"id": "dbn_feature", "name": "Feature Extraction"},
            {"id": "dbn_pretraining", "name": "Pre-training"}
        ],
        "pinn": [
            {"id": "pinn_differential", "name": "Differential Equation Solving"},
            {"id": "pinn_simulation", "name": "Physics Simulation"}
        ],
        "neural_ode": [
            {"id": "neural_ode_timeseries", "name": "Time Series"},
            {"id": "neural_ode_dynamics", "name": "Dynamic Systems"}
        ],
        "gnn": [
            {"id": "gnn_node", "name": "Node Classification"},
            {"id": "gnn_edge", "name": "Edge Classification"}
        ],
        "snn": [
            {"id": "snn_neuromorphic", "name": "Neuromorphic Computing"},
            {"id": "snn_event", "name": "Event-driven Systems"}
        ]
    }
    
    return templates.TemplateResponse(
        "neural_network_detail.html",
        {
            "request": request,
            "user": user,
            "tasks": network_tasks.get(network_id, []),
            **NEURAL_NETWORK_DATA[network_id]
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
