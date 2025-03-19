from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

# Database configuration
POSTGRESQL_DBNAME = os.getenv('POSTGRESQL_DBNAME')
POSTGRESQL_USER = os.getenv('POSTGRESQL_USER')
POSTGRESQL_PASSWORD = os.getenv('POSTGRESQL_PASSWORD')
POSTGRESQL_HOST = os.getenv('POSTGRESQL_HOST')
POSTGRESQL_PORT = os.getenv('POSTGRESQL_PORT')

DATABASE_URL = f"postgresql://{POSTGRESQL_USER}:{POSTGRESQL_PASSWORD}@{POSTGRESQL_HOST}:{POSTGRESQL_PORT}/{POSTGRESQL_DBNAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class InvoiceItem(Base):
    __tablename__ = "invoice_items"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey('invoice_details.document_id'))
    item_name = Column(String)
    quantity = Column(Float)
    unit_price = Column(Float)
    total_price = Column(Float)
    description = Column(Text, nullable=True)

class TaxDetails(Base):
    __tablename__ = "tax_details"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey('invoice_details.document_id'))
    tax_type = Column(String)  # e.g., "GST", "VAT", etc.
    tax_rate = Column(Float)
    tax_amount = Column(Float)
    description = Column(Text, nullable=True)

class InvoiceDetails(Base):
    __tablename__ = "invoice_details"

    document_id = Column(String, primary_key=True)
    generated_name = Column(String)
    invoice_number = Column(String)
    invoice_date = Column(DateTime)
    due_date = Column(DateTime)
    biller_company = Column(String)
    biller_address = Column(Text)
    billed_to_company = Column(String)
    billed_to_address = Column(Text)
    total_amount = Column(Float)
    currency = Column(String)
    payment_terms = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Add relationships
    items = relationship("InvoiceItem", cascade="all, delete-orphan")
    tax_details = relationship("TaxDetails", cascade="all, delete-orphan")
    subtotal = Column(Float)
    total_tax = Column(Float)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 