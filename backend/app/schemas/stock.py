from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StockBase(BaseModel):
    symbol: str = Field(min_length=1, max_length=30)
    company_name: str = Field(min_length=2, max_length=150)
    exchange: str = Field(default="NSE", max_length=10)
    instrument_key: str = Field(min_length=3, max_length=100)
    sector: str | None = Field(default=None, max_length=100)

    @field_validator("symbol", "exchange", mode="before")
    @classmethod
    def convert_to_uppercase(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("company_name", "instrument_key", "sector", mode="before")
    @classmethod
    def remove_extra_spaces(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned_value = value.strip()
        return cleaned_value or None


class StockCreate(StockBase):
    pass


class StockUpdate(BaseModel):
    company_name: str | None = Field(default=None, min_length=2, max_length=150)
    instrument_key: str | None = Field(default=None, min_length=3, max_length=100)
    sector: str | None = Field(default=None, max_length=100)
    is_active: bool | None = None

    @field_validator(
        "company_name",
        "instrument_key",
        "sector",
        mode="before",
    )
    @classmethod
    def remove_extra_spaces(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned_value = value.strip()
        return cleaned_value or None


class StockResponse(StockBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime