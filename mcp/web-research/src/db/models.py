"""SQLAlchemy ORM models for domain tracking and telemetry"""

from datetime import datetime
from typing import Optional
from sqlalchemy import String, Boolean, Integer, DateTime, Float, func, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models"""
    pass


class Domain(Base):
    """Domain tracking model"""
    __tablename__ = "domains"

    domain: Mapped[str] = mapped_column(String(255), primary_key=True)
    preferred_method: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="crawl4ai",
        server_default="crawl4ai"
    )
    last_success: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_failure: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_blacklisted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

    # Indexes
    __table_args__ = (
        Index("idx_domains_blacklisted", "is_blacklisted"),
        Index("idx_domains_method", "preferred_method", "is_blacklisted"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "domain": self.domain,
            "preferred_method": self.preferred_method,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "failure_count": self.failure_count,
            "is_blacklisted": self.is_blacklisted,
        }


class ScrapeMetric(Base):
    """Telemetry model for tracking scrape request metrics"""
    __tablename__ = "scrape_metrics"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    method: Mapped[str] = mapped_column(String(50), nullable=False)  # crawl4ai, selenium, pdf, reddit_api
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)  # Duration in milliseconds
    content_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Characters returned
    error: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # Error message if failed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True
    )

    # Indexes for common queries
    __table_args__ = (
        Index("idx_metrics_domain_created", "domain", "created_at"),
        Index("idx_metrics_success_created", "success", "created_at"),
        Index("idx_metrics_method_created", "method", "created_at"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "url": self.url,
            "domain": self.domain,
            "method": self.method,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "content_length": self.content_length,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
