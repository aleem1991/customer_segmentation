from pydantic import BaseModel, Field
from typing import List, Dict, Any

class CustomerInput(BaseModel):
    recency: int = Field(..., description="Days since last purchase", ge=0, json_schema_extra={"example": 65})
    frequency: int = Field(..., description="Number of unique orders/invoices", ge=1, json_schema_extra={"example": 3})
    monetary: float = Field(..., description="Average value spend per invoice", ge=0.0, json_schema_extra={"example": 350.50})
    basket_size: float = Field(..., description="Average items count per basket size", ge=0.0, json_schema_extra={"example": 12.5})
    avg_days_between: float | None = Field(None, description="Average days between purchases.", ge=0.0)
    recent_orders_ratio: float | None = Field(None, description="Ratio of orders placed in last 60 days.", ge=0.0, le=1.0)
    is_uk: int | None = Field(None, description="1 if customer is in the UK, else 0.", ge=0, le=1)

class BatchInput(BaseModel):
    customers: List[CustomerInput]
