# API Documentation

> **Customer Churn Prediction & Retention Optimization System — REST API Reference**

This document covers all REST endpoints exposed by the FastAPI-based services:
real-time churn scoring and personalized recommendation.

The API server runs inside the `pipeline` Docker container and is accessible
at **`http://localhost:8000`** by default. Interactive docs are available at
`/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Common Headers & Error Responses](#2-common-headers--error-responses)
3. [Health & Readiness](#3-health--readiness)
4. [Real-Time Churn Scoring](#4-real-time-churn-scoring)
   - 4.1 [Score a Single Customer](#41-score-a-single-customer)
   - 4.2 [Batch Score Multiple Customers](#42-batch-score-multiple-customers)
   - 4.3 [Stream Event for Real-Time Update](#43-stream-event-for-real-time-update)
5. [Personalized Recommendations](#5-personalized-recommendations)
   - 5.1 [Get Recommendations for a Customer](#51-get-recommendations-for-a-customer)
   - 5.2 [Batch Recommendations](#52-batch-recommendations)
6. [Customer Segments & Uplift](#6-customer-segments--uplift)
   - 6.1 [Get Customer Segment Info](#61-get-customer-segment-info)
   - 6.2 [List Retention Targets](#62-list-retention-targets)
7. [Model Metadata](#7-model-metadata)
   - 7.1 [Get Active Model Info](#71-get-active-model-info)
   - 7.2 [Get Feature Importance](#72-get-feature-importance)
8. [Configuration Reference](#8-configuration-reference)

---

## 1. Authentication

The API uses **API Key** authentication via the `X-API-Key` HTTP header.
The key is configured in `config/api_config.yaml` under `api.auth.api_key`.

| Header       | Type   | Required | Description          |
|-------------|--------|----------|----------------------|
| `X-API-Key` | string | Yes      | API key for access   |

In development/local Docker Compose environments, the default key is
`churn-api-dev-key-2024`. **Change this in production.**

**Example — authenticated request:**

```bash
curl -H "X-API-Key: churn-api-dev-key-2024" http://localhost:8000/api/v1/health
```

**Unauthorized response (401):**

```json
{
  "detail": "Invalid or missing API key"
}
```

---

## 2. Common Headers & Error Responses

### Request Headers

| Header         | Type   | Required | Description                     |
|---------------|--------|----------|---------------------------------|
| `X-API-Key`   | string | Yes      | Authentication key              |
| `Content-Type` | string | Yes (POST) | Must be `application/json`   |

### Standard Error Response Schema

```json
{
  "detail": "string — human-readable error message",
  "error_code": "string — machine-readable error code",
  "timestamp": "2024-06-15T10:30:00Z"
}
```

### HTTP Status Codes

| Code | Meaning                        |
|------|--------------------------------|
| 200  | Success                        |
| 201  | Resource created               |
| 400  | Bad request / validation error |
| 401  | Unauthorized — invalid API key |
| 404  | Customer or resource not found |
| 422  | Unprocessable entity           |
| 429  | Rate limit exceeded            |
| 500  | Internal server error          |
| 503  | Model not loaded / service unavailable |

---

## 3. Health & Readiness

### `GET /api/v1/health`

Returns service health status and loaded model information.

**Response (200):**

```json
{
  "status": "healthy",
  "timestamp": "2024-06-15T10:30:00Z",
  "version": "1.0.0",
  "models_loaded": {
    "ml_model": true,
    "dl_model": true,
    "recommendation_model": true,
    "survival_model": true
  },
  "redis_connected": true,
  "feature_store_available": true
}
```

**curl:**

```bash
curl -s -H "X-API-Key: churn-api-dev-key-2024" \
  http://localhost:8000/api/v1/health | python -m json.tool
```

---

### `GET /api/v1/readiness`

Kubernetes-style readiness probe. Returns 200 only when all models are
loaded and dependencies are available.

**Response (200):**

```json
{
  "ready": true
}
```

**Response (503) — not ready:**

```json
{
  "ready": false,
  "reason": "ML model not loaded"
}
```

---

## 4. Real-Time Churn Scoring

### 4.1 Score a Single Customer

#### `POST /api/v1/scoring/predict`

Computes the churn probability for a single customer using the ensemble
model (ML weight 0.6 + DL weight 0.4, configurable).

**Request Body:**

| Field         | Type   | Required | Description                                    |
|--------------|--------|----------|------------------------------------------------|
| `customer_id` | string | Yes      | Unique customer identifier (e.g., `"C00847"`)  |
| `use_cache`   | bool   | No       | Use Redis-cached features (default: `true`)    |
| `model_type`  | string | No       | `"ensemble"` (default), `"ml"`, or `"dl"`      |

**Request Example:**

```json
{
  "customer_id": "C00847",
  "use_cache": true,
  "model_type": "ensemble"
}
```

**Response (200):**

```json
{
  "customer_id": "C00847",
  "churn_probability": 0.7832,
  "churn_risk_level": "high",
  "model_type": "ensemble",
  "ml_score": 0.8105,
  "dl_score": 0.7422,
  "ensemble_weights": {"ml": 0.6, "dl": 0.4},
  "top_risk_factors": [
    {"feature": "days_since_last_purchase", "value": 45, "importance": 0.182},
    {"feature": "purchase_frequency_change_4w", "value": -0.65, "importance": 0.156},
    {"feature": "session_duration_trend", "value": -0.43, "importance": 0.134}
  ],
  "segment": "high_value_persuadable",
  "recommended_action": "VIP Care 20% Coupon",
  "scored_at": "2024-06-15T10:30:05Z",
  "latency_ms": 23
}
```

**Risk Levels:**

| Level    | Probability Range |
|----------|-------------------|
| `low`    | 0.0 – 0.3        |
| `medium` | 0.3 – 0.6        |
| `high`   | 0.6 – 0.8        |
| `critical` | 0.8 – 1.0      |

**curl:**

```bash
curl -s -X POST http://localhost:8000/api/v1/scoring/predict \
  -H "X-API-Key: churn-api-dev-key-2024" \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "C00847", "model_type": "ensemble"}' | python -m json.tool
```

---

### 4.2 Batch Score Multiple Customers

#### `POST /api/v1/scoring/predict/batch`

Scores multiple customers in a single request. Maximum batch size is
configurable (default: 1000).

**Request Body:**

| Field          | Type     | Required | Description                          |
|---------------|----------|----------|--------------------------------------|
| `customer_ids` | string[] | Yes      | List of customer IDs                 |
| `model_type`   | string   | No       | `"ensemble"` (default), `"ml"`, `"dl"` |
| `include_factors` | bool  | No       | Include top risk factors (default: `false`) |

**Request Example:**

```json
{
  "customer_ids": ["C00847", "C01293", "C00562"],
  "model_type": "ensemble",
  "include_factors": true
}
```

**Response (200):**

```json
{
  "results": [
    {
      "customer_id": "C00847",
      "churn_probability": 0.7832,
      "churn_risk_level": "high",
      "segment": "high_value_persuadable"
    },
    {
      "customer_id": "C01293",
      "churn_probability": 0.6541,
      "churn_risk_level": "high",
      "segment": "high_value_persuadable"
    },
    {
      "customer_id": "C00562",
      "churn_probability": 0.3210,
      "churn_risk_level": "medium",
      "segment": "low_value_sure_thing"
    }
  ],
  "total": 3,
  "scored_at": "2024-06-15T10:30:10Z",
  "model_type": "ensemble",
  "latency_ms": 45
}
```

**curl:**

```bash
curl -s -X POST http://localhost:8000/api/v1/scoring/predict/batch \
  -H "X-API-Key: churn-api-dev-key-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_ids": ["C00847", "C01293", "C00562"],
    "model_type": "ensemble",
    "include_factors": true
  }' | python -m json.tool
```

---

### 4.3 Stream Event for Real-Time Update

#### `POST /api/v1/scoring/event`

Ingests a single customer event into the Redis Stream for real-time
feature update and churn score recalculation. The scoring pipeline
consumes events asynchronously and updates cached scores.

**Request Body:**

| Field         | Type   | Required | Description                                      |
|--------------|--------|----------|--------------------------------------------------|
| `customer_id` | string | Yes      | Customer identifier                              |
| `event_type`  | string | Yes      | One of: `page_view`, `search`, `add_to_cart`, `remove_from_cart`, `purchase`, `coupon_use`, `review`, `cs_contact` |
| `timestamp`   | string | No       | ISO 8601 datetime (default: current time)        |
| `properties`  | object | No       | Additional event properties                       |

**Request Example:**

```json
{
  "customer_id": "C00847",
  "event_type": "purchase",
  "timestamp": "2024-06-15T10:28:00Z",
  "properties": {
    "order_value": 85000,
    "items_count": 3,
    "used_coupon": true,
    "coupon_id": "WINBACK15"
  }
}
```

**Response (201):**

```json
{
  "status": "accepted",
  "event_id": "evt_1718444880_C00847",
  "stream_id": "1718444880123-0",
  "message": "Event queued for real-time processing"
}
```

**curl:**

```bash
curl -s -X POST http://localhost:8000/api/v1/scoring/event \
  -H "X-API-Key: churn-api-dev-key-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C00847",
    "event_type": "purchase",
    "properties": {"order_value": 85000, "used_coupon": true}
  }' | python -m json.tool
```

---

## 5. Personalized Recommendations

### 5.1 Get Recommendations for a Customer

#### `GET /api/v1/recommendations/{customer_id}`

Returns personalized product/action recommendations for an at-risk
customer, combining churn risk, purchase history, and uplift score.

**Path Parameters:**

| Parameter     | Type   | Description            |
|--------------|--------|------------------------|
| `customer_id` | string | Customer identifier    |

**Query Parameters:**

| Parameter  | Type | Required | Description                          |
|-----------|------|----------|--------------------------------------|
| `top_n`    | int  | No       | Number of recommendations (default: 5, max: 20) |
| `strategy` | str  | No       | `"retention"` (default), `"upsell"`, `"reactivation"` |

**Response (200):**

```json
{
  "customer_id": "C00847",
  "churn_probability": 0.7832,
  "segment": "high_value_persuadable",
  "clv_predicted": 2450000,
  "recommendations": [
    {
      "rank": 1,
      "type": "coupon",
      "action": "VIP 20% 할인 쿠폰",
      "description": "최근 관심 카테고리(전자기기) 20% 할인 쿠폰 발송",
      "expected_uplift": 0.152,
      "estimated_cost_krw": 30000,
      "confidence": 0.89,
      "category": "electronics",
      "reason": "High purchase history in electronics with declining frequency"
    },
    {
      "rank": 2,
      "type": "push_notification",
      "action": "개인화 푸시 알림",
      "description": "위시리스트 상품 가격 인하 알림",
      "expected_uplift": 0.098,
      "estimated_cost_krw": 500,
      "confidence": 0.75,
      "category": "wishlist",
      "reason": "3 items in wishlist with recent price drops"
    },
    {
      "rank": 3,
      "type": "email",
      "action": "VIP 전용 프리뷰 초대",
      "description": "신상품 프리뷰 이벤트 초대 이메일",
      "expected_uplift": 0.067,
      "estimated_cost_krw": 200,
      "confidence": 0.62,
      "category": "engagement",
      "reason": "Historically responsive to exclusive events"
    }
  ],
  "total_estimated_cost_krw": 30700,
  "total_expected_uplift": 0.317,
  "generated_at": "2024-06-15T10:31:00Z"
}
```

**curl:**

```bash
curl -s -H "X-API-Key: churn-api-dev-key-2024" \
  "http://localhost:8000/api/v1/recommendations/C00847?top_n=5&strategy=retention" \
  | python -m json.tool
```

---

### 5.2 Batch Recommendations

#### `POST /api/v1/recommendations/batch`

Generate recommendations for multiple customers at once.

**Request Body:**

| Field          | Type     | Required | Description                        |
|---------------|----------|----------|------------------------------------|
| `customer_ids` | string[] | Yes      | List of customer IDs (max: 100)   |
| `top_n`        | int      | No       | Recommendations per customer (default: 3) |
| `strategy`     | string   | No       | `"retention"`, `"upsell"`, `"reactivation"` |

**Request Example:**

```json
{
  "customer_ids": ["C00847", "C01293"],
  "top_n": 3,
  "strategy": "retention"
}
```

**Response (200):**

```json
{
  "results": [
    {
      "customer_id": "C00847",
      "churn_probability": 0.7832,
      "recommendations": [
        {"rank": 1, "type": "coupon", "action": "VIP 20% 할인 쿠폰", "expected_uplift": 0.152}
      ]
    },
    {
      "customer_id": "C01293",
      "churn_probability": 0.6541,
      "recommendations": [
        {"rank": 1, "type": "push_notification", "action": "장바구니 리마인더", "expected_uplift": 0.121}
      ]
    }
  ],
  "total_customers": 2,
  "generated_at": "2024-06-15T10:32:00Z"
}
```

**curl:**

```bash
curl -s -X POST http://localhost:8000/api/v1/recommendations/batch \
  -H "X-API-Key: churn-api-dev-key-2024" \
  -H "Content-Type: application/json" \
  -d '{"customer_ids": ["C00847", "C01293"], "top_n": 3}' | python -m json.tool
```

---

## 6. Customer Segments & Uplift

### 6.1 Get Customer Segment Info

#### `GET /api/v1/customers/{customer_id}/segment`

Returns detailed segment classification for a customer including
churn probability, uplift score, CLV, and priority score.

**Response (200):**

```json
{
  "customer_id": "C00847",
  "churn_probability": 0.7832,
  "uplift_score": 0.152,
  "clv_predicted": 2450000,
  "priority_score": 372600.0,
  "uplift_quadrant": "persuadable",
  "segment_6": "high_value_persuadable",
  "segment_label": "고가치-Persuadable (최우선 리텐션 대상)",
  "persona": "high_value_at_risk",
  "retention_strategy": {
    "name": "VIP Care Package",
    "actions": ["20% 할인 쿠폰", "전담 CS 매니저 배정", "프리뷰 초대"],
    "estimated_cost_krw": 70000,
    "expected_retention_lift": 0.15
  },
  "survival_analysis": {
    "median_survival_days": 42,
    "30_day_survival_prob": 0.35,
    "60_day_survival_prob": 0.12,
    "hazard_ratio": 2.31
  },
  "updated_at": "2024-06-15T10:30:00Z"
}
```

**curl:**

```bash
curl -s -H "X-API-Key: churn-api-dev-key-2024" \
  http://localhost:8000/api/v1/customers/C00847/segment | python -m json.tool
```

---

### 6.2 List Retention Targets

#### `GET /api/v1/segments/retention-targets`

Returns a paginated list of customers prioritized for retention
intervention, sorted by priority score (uplift × CLV) descending.

**Query Parameters:**

| Parameter    | Type   | Required | Description                                   |
|-------------|--------|----------|-----------------------------------------------|
| `segment`    | string | No       | Filter by segment: `high_value_persuadable`, `low_value_persuadable`, `high_value_sure_thing`, `high_value_lost_cause`, `low_value_lost_cause`, `new_customer_onboarding` |
| `min_churn`  | float  | No       | Minimum churn probability filter (0.0–1.0)    |
| `min_uplift` | float  | No       | Minimum uplift score filter                    |
| `page`       | int    | No       | Page number (default: 1)                       |
| `page_size`  | int    | No       | Results per page (default: 50, max: 200)       |

**Response (200):**

```json
{
  "targets": [
    {
      "rank": 1,
      "customer_id": "C00847",
      "churn_probability": 0.89,
      "uplift_score": 0.152,
      "clv_predicted": 2450000,
      "priority_score": 372600.0,
      "segment": "high_value_persuadable",
      "recommended_action": "VIP Care 20%"
    },
    {
      "rank": 2,
      "customer_id": "C01293",
      "churn_probability": 0.82,
      "uplift_score": 0.148,
      "clv_predicted": 1890000,
      "priority_score": 279720.0,
      "segment": "high_value_persuadable",
      "recommended_action": "VIP Care 20%"
    }
  ],
  "total": 3847,
  "page": 1,
  "page_size": 50,
  "total_pages": 77
}
```

**curl:**

```bash
curl -s -H "X-API-Key: churn-api-dev-key-2024" \
  "http://localhost:8000/api/v1/segments/retention-targets?segment=high_value_persuadable&min_churn=0.7&page=1&page_size=10" \
  | python -m json.tool
```

---

## 7. Model Metadata

### 7.1 Get Active Model Info

#### `GET /api/v1/models/info`

Returns metadata about the currently loaded models including version,
training metrics, and MLflow run IDs.

**Response (200):**

```json
{
  "models": {
    "ml": {
      "name": "LightGBM",
      "version": "1.0.0",
      "mlflow_run_id": "a1b2c3d4e5f6",
      "trained_at": "2024-06-15T08:00:00Z",
      "metrics": {
        "auc_roc": 0.852,
        "precision": 0.74,
        "recall": 0.71,
        "f1_score": 0.72
      },
      "feature_count": 35,
      "training_samples": 16000,
      "cross_validation_folds": 5
    },
    "dl": {
      "name": "Transformer",
      "version": "1.0.0",
      "mlflow_run_id": "g7h8i9j0k1l2",
      "trained_at": "2024-06-15T08:30:00Z",
      "metrics": {
        "auc_roc": 0.839,
        "precision": 0.71,
        "recall": 0.73,
        "f1_score": 0.72
      },
      "sequence_length": 50,
      "embedding_dim": 64
    },
    "ensemble": {
      "weights": {"ml": 0.6, "dl": 0.4},
      "metrics": {
        "auc_roc": 0.861,
        "precision": 0.76,
        "recall": 0.74,
        "f1_score": 0.75
      }
    },
    "survival": {
      "name": "CoxPH",
      "version": "1.0.0",
      "concordance_index": 0.812
    }
  },
  "churn_definition": {
    "no_purchase_days": 30,
    "no_login_days": 60,
    "operator": "OR"
  },
  "last_updated": "2024-06-15T08:35:00Z"
}
```

**curl:**

```bash
curl -s -H "X-API-Key: churn-api-dev-key-2024" \
  http://localhost:8000/api/v1/models/info | python -m json.tool
```

---

### 7.2 Get Feature Importance

#### `GET /api/v1/models/feature-importance`

Returns global SHAP-based feature importance for the ML model.

**Query Parameters:**

| Parameter | Type | Required | Description                       |
|----------|------|----------|-----------------------------------|
| `top_n`   | int  | No       | Number of features (default: 10)  |

**Response (200):**

```json
{
  "model": "LightGBM",
  "method": "SHAP",
  "features": [
    {"rank": 1, "name": "days_since_last_purchase", "importance": 0.182},
    {"rank": 2, "name": "purchase_frequency_change_4w", "importance": 0.156},
    {"rank": 3, "name": "session_duration_trend", "importance": 0.134},
    {"rank": 4, "name": "cart_abandonment_rate", "importance": 0.098},
    {"rank": 5, "name": "coupon_response_rate_change", "importance": 0.087},
    {"rank": 6, "name": "recency_days", "importance": 0.076},
    {"rank": 7, "name": "avg_session_minutes", "importance": 0.065},
    {"rank": 8, "name": "purchase_cycle_anomaly", "importance": 0.058},
    {"rank": 9, "name": "weekend_purchase_ratio", "importance": 0.045},
    {"rank": 10, "name": "rfm_monetary", "importance": 0.041}
  ],
  "total_features": 35
}
```

**curl:**

```bash
curl -s -H "X-API-Key: churn-api-dev-key-2024" \
  "http://localhost:8000/api/v1/models/feature-importance?top_n=10" | python -m json.tool
```

---

## 8. Configuration Reference

All API settings are managed via `config/api_config.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 2
  title: "Churn Prediction & Retention API"
  version: "1.0.0"

  auth:
    api_key: "churn-api-dev-key-2024"    # Change in production
    rate_limit_per_minute: 120

  scoring:
    default_model_type: "ensemble"
    ensemble_weights:
      ml: 0.6
      dl: 0.4
    risk_thresholds:
      low: 0.3
      medium: 0.6
      high: 0.8
    batch_max_size: 1000
    cache_ttl_seconds: 300               # Redis cache TTL for scores

  recommendations:
    default_top_n: 5
    max_top_n: 20
    strategies:
      - retention
      - upsell
      - reactivation

  redis:
    host: "redis"
    port: 6379
    db: 0
    stream_key: "customer_events"
    score_cache_prefix: "score:"
    feature_cache_prefix: "features:"

  mlflow:
    tracking_uri: "http://mlflow:5000"
```

### Environment Variable Overrides

| Variable           | Description                        | Default              |
|--------------------|------------------------------------|----------------------|
| `API_HOST`         | Bind address                       | `0.0.0.0`           |
| `API_PORT`         | Bind port                          | `8000`               |
| `API_KEY`          | Override authentication key         | from config YAML     |
| `REDIS_HOST`       | Redis hostname                     | `redis`              |
| `REDIS_PORT`       | Redis port                         | `6379`               |
| `MLFLOW_TRACKING_URI` | MLflow server URL              | `http://mlflow:5000` |

---

## Appendix A: Pydantic Request/Response Schemas

Below are the full Pydantic model definitions used by the API.

### Scoring Schemas

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ModelType(str, Enum):
    ensemble = "ensemble"
    ml = "ml"
    dl = "dl"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ScoreRequest(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier", example="C00847")
    use_cache: bool = Field(True, description="Use Redis-cached features")
    model_type: ModelType = Field(ModelType.ensemble, description="Model to use for prediction")


class RiskFactor(BaseModel):
    feature: str
    value: float
    importance: float


class ScoreResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_risk_level: RiskLevel
    model_type: ModelType
    ml_score: Optional[float] = None
    dl_score: Optional[float] = None
    ensemble_weights: Optional[dict] = None
    top_risk_factors: Optional[List[RiskFactor]] = None
    segment: Optional[str] = None
    recommended_action: Optional[str] = None
    scored_at: datetime
    latency_ms: int


class BatchScoreRequest(BaseModel):
    customer_ids: List[str] = Field(..., max_length=1000)
    model_type: ModelType = Field(ModelType.ensemble)
    include_factors: bool = Field(False)


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    total: int
    scored_at: datetime
    model_type: ModelType
    latency_ms: int
```

### Event Streaming Schemas

```python
class EventType(str, Enum):
    page_view = "page_view"
    search = "search"
    add_to_cart = "add_to_cart"
    remove_from_cart = "remove_from_cart"
    purchase = "purchase"
    coupon_use = "coupon_use"
    review = "review"
    cs_contact = "cs_contact"


class EventRequest(BaseModel):
    customer_id: str = Field(..., example="C00847")
    event_type: EventType
    timestamp: Optional[datetime] = None
    properties: Optional[dict] = Field(None, example={"order_value": 85000})


class EventResponse(BaseModel):
    status: str = "accepted"
    event_id: str
    stream_id: str
    message: str
```

### Recommendation Schemas

```python
class RecommendationStrategy(str, Enum):
    retention = "retention"
    upsell = "upsell"
    reactivation = "reactivation"


class Recommendation(BaseModel):
    rank: int
    type: str
    action: str
    description: str
    expected_uplift: float
    estimated_cost_krw: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: str
    reason: str


class RecommendationResponse(BaseModel):
    customer_id: str
    churn_probability: float
    segment: str
    clv_predicted: float
    recommendations: List[Recommendation]
    total_estimated_cost_krw: int
    total_expected_uplift: float
    generated_at: datetime


class BatchRecommendationRequest(BaseModel):
    customer_ids: List[str] = Field(..., max_length=100)
    top_n: int = Field(3, ge=1, le=20)
    strategy: RecommendationStrategy = Field(RecommendationStrategy.retention)
```

---

## Appendix B: Complete Endpoint Summary

| Method | Endpoint                                   | Description                        |
|--------|--------------------------------------------|------------------------------------|
| GET    | `/api/v1/health`                           | Health check                       |
| GET    | `/api/v1/readiness`                        | Readiness probe                    |
| POST   | `/api/v1/scoring/predict`                  | Score single customer              |
| POST   | `/api/v1/scoring/predict/batch`            | Batch score customers              |
| POST   | `/api/v1/scoring/event`                    | Ingest streaming event             |
| GET    | `/api/v1/recommendations/{customer_id}`    | Get personalized recommendations   |
| POST   | `/api/v1/recommendations/batch`            | Batch recommendations              |
| GET    | `/api/v1/customers/{customer_id}/segment`  | Get customer segment detail        |
| GET    | `/api/v1/segments/retention-targets`       | List prioritized retention targets |
| GET    | `/api/v1/models/info`                      | Active model metadata              |
| GET    | `/api/v1/models/feature-importance`        | SHAP feature importance            |

---

## Appendix C: Rate Limiting

The API enforces rate limits per API key:

- **Default**: 120 requests per minute
- **Batch endpoints**: Each customer in a batch counts as 1 request
- Rate limit headers are included in every response:

| Header                  | Description                    |
|------------------------|--------------------------------|
| `X-RateLimit-Limit`    | Maximum requests per window    |
| `X-RateLimit-Remaining`| Remaining requests             |
| `X-RateLimit-Reset`    | Unix timestamp of window reset |

**Rate limit exceeded (429):**

```json
{
  "detail": "Rate limit exceeded. Retry after 23 seconds.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 23
}
```

---

## Appendix D: Docker Compose Network

Within the Docker Compose setup, the API is accessible:

- **From host machine**: `http://localhost:8000`
- **From other containers**: `http://pipeline:8000`
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

The Streamlit dashboard (`http://localhost:8501`) communicates with the
API internally via `http://pipeline:8000`.
