from fastapi import FastAPI
from .hyperparameters import router as hyperparameters_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Neural Network API",
        description="API for managing neural network hyperparameters and training",
        version="1.0.0"
    )
    
    # Include the hyperparameters router
    app.include_router(hyperparameters_router)
    
    return app 