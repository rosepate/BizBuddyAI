from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class SalesRecord(BaseModel):
       transaction_id: str = Field(..., description="Unique transaction identifier")
       date: datetime = Field(..., description="Date of the sale")
       product: str = Field(..., description="Name of the product")
       category: str = Field(..., description="Product category")
       units_sold: int = Field(..., ge=0, description="Number of units sold")
       inventory_after: int = Field(..., ge=0, description="Inventory remaining after sale")
       location: str = Field(..., description="Store location")
       platform: str = Field(..., description="Sales platform")
       payment_method: str = Field(..., description="Payment method used")
       expiry_date: Optional[datetime] = Field(None, description="Product expiry date")
       unit_price: float = Field(..., ge=0.0, description="Price per unit")
       cost_price: float = Field(..., ge=0.0, description="Cost per unit")
       revenue: float = Field(..., ge=0.0, description="Total revenue from sale")
       profit: float = Field(..., description="Profit from sale")