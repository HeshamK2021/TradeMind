from __future__ import annotations
from uuid import uuid4
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

def _uuid() -> str:
    return str(uuid4())

class Candle(Base):
    __tablename__ = "candle"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    tf: Mapped[str] = mapped_column(String(12), nullable=False)
    ts: Mapped[int] = mapped_column(Integer, nullable=False) 
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    __table_args__ = (
        UniqueConstraint("symbol", "tf", "ts", name="uq_candle_symbol_tf_ts"),
        Index("ix_candle_symbol_tf_ts", "symbol", "tf", "ts"),
        CheckConstraint(
            "open >= 0 AND high >= 0 AND low >= 0 AND close >= 0 AND volume >= 0",
            name="ck_nonnegative",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Candle {self.symbol} {self.tf} ts={self.ts} "
            f"ohlc=({self.open},{self.high},{self.low},{self.close}) vol={self.volume}>"
        )

class FeatureRow(Base):
    __tablename__ = "feature_row"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    tf: Mapped[str] = mapped_column(String(12), nullable=False)

    ts: Mapped[int] = mapped_column(Integer, nullable=False)  

    ema_5: Mapped[float] = mapped_column(Float, nullable=False)
    ema_20: Mapped[float] = mapped_column(Float, nullable=False)
    rsi_14: Mapped[float] = mapped_column(Float, nullable=False)
    atr_14: Mapped[float] = mapped_column(Float, nullable=False)
    bb_mid: Mapped[float] = mapped_column(Float, nullable=False)
    bb_up: Mapped[float] = mapped_column(Float, nullable=False)
    bb_dn: Mapped[float] = mapped_column(Float, nullable=False)

    regime_flag: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    quality: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)

    shifted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (
        UniqueConstraint("symbol", "tf", "ts", name="uq_feature_symbol_tf_ts"),
        Index("ix_feature_symbol_tf_ts", "symbol", "tf", "ts"),
    )

    def __repr__(self) -> str:
        return (
            f"<FeatureRow {self.symbol} {self.tf} ts={self.ts} "
            f"ema5={self.ema_5:.4f} ema20={self.ema_20:.4f} rsi={self.rsi_14:.2f}>"
        )

class BacktestRun(Base):
    __tablename__ = "backtest_run"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    created_at: Mapped[Optional[str]] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    params_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    equity_curve_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    def __repr__(self) -> str:
        return f"<BacktestRun id={self.id} created_at={self.created_at}>"

class Recommendation(Base):
    __tablename__ = "recommendation"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    ts: Mapped[int] = mapped_column(Integer, nullable=False)

    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    tf: Mapped[str] = mapped_column(String(12), nullable=False)
    action: Mapped[str] = mapped_column(String(8), nullable=False)  
