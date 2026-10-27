"""YieldSense FastAPI application: Production REST API for NAND test yield analytics."""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.api.dependencies import ModelManager, get_model_manager
from src.api.pipeline import process_batch_lots, process_single_lot
from src.api.schemas import (
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    HealthResponse,
    LotAnalysisRequest,
    LotAnalysisResponse,
    ModelInfoResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload model, scaler, and configuration upon service startup."""
    manager = get_model_manager()
    try:
        manager.load_artifacts()
    except Exception as e:
        print(f"Warning during artifact load on startup: {e}")
    yield


app = FastAPI(
    title="YieldSense REST API",
    description="AI-Enabled NAND Test Yield & Recovery Analytics Pipeline",
    version=__version__,
    lifespan=lifespan,
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Return clean HTTP 422 for input validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError):
    """Return clean HTTP 503 for uninitialized model states."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    manager: ModelManager = Depends(get_model_manager),
) -> HealthResponse:
    """Return service health status and artifact readiness."""
    return HealthResponse(
        status="HEALTHY" if manager.is_ready else "DEGRADED",
        model_loaded=manager.model is not None,
        scaler_loaded=manager.scaler is not None,
        expected_sensors_count=len(manager.expected_sensors),
        version=__version__,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info(
    manager: ModelManager = Depends(get_model_manager),
) -> ModelInfoResponse:
    """Return active ML model type, evaluation metrics, and hyperparameter configuration."""
    if not manager.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded.",
        )

    return ModelInfoResponse(
        model_type="XGBClassifier",
        evaluation_metrics=manager.eval_metrics.get("models", {}).get(
            "xgboost_tuned", {}
        ),
        best_hyperparameters=manager.best_params,
    )


@app.post("/analyze-lot", response_model=LotAnalysisResponse, tags=["Analytics"])
async def analyze_lot(
    request: LotAnalysisRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> LotAnalysisResponse:
    """Analyze a single lot's sensor readings and return prediction, SHAP attribution, and recovery score."""
    try:
        return process_single_lot(request, manager)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )


@app.post("/batch-analyze", response_model=BatchAnalysisResponse, tags=["Analytics"])
async def batch_analyze(
    request: BatchAnalysisRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> BatchAnalysisResponse:
    """Analyze a batch of lots and return ranked recovery candidates with aggregate summary."""
    try:
        return process_batch_lots(request, manager)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
