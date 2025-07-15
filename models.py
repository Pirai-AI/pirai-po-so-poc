from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, ForeignKey, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, date
import os
from dotenv import load_dotenv
import uuid
from sqlalchemy.orm import Session

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
    invoice_date = Column(Date)
    due_date = Column(Date)
    biller_company = Column(String)
    biller_address = Column(Text)
    recipient_name = Column(String)
    recipient_address = Column(Text)
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

def save_invoice_details(document_id: str, details: dict, db: Session):
    """Save invoice details to database"""
    try:
        # Parse dates with fallback to default date if 'N/A'
        def parse_date(date_str):
            if date_str == 'N/A' or not date_str:
                return date(2000, 1, 1)  # Default date
            try:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return date(2000, 1, 1)  # Default date on parse error

        invoice_details = InvoiceDetails(
            document_id=document_id,
            generated_name=details.get('generated_name'),
            invoice_number=details.get('invoice_number', 'N/A'),
            invoice_date=parse_date(details.get('invoice_date')),
            due_date=parse_date(details.get('due_date')),
            biller_company=details.get('biller_company', 'N/A'),
            biller_address=details.get('biller_address', 'N/A'),
            recipient_name=details.get('recipient_name', 'N/A'),
            recipient_address=details.get('recipient_address', 'N/A'),
            subtotal=details.get('subtotal', 0),
            total_tax=details.get('total_tax', 0),
            total_amount=details.get('total_amount', 0),
            currency=details.get('currency', 'USD'),
            payment_terms=details.get('payment_terms', 'N/A')
        )
        
        # Add items
        for item in details.get('items', []):
            invoice_item = InvoiceItem(
                document_id=document_id,
                item_name=item.get('item_name', 'N/A'),
                quantity=float(item.get('quantity', 0)),
                unit_price=float(item.get('unit_price', 0)),
                total_price=float(item.get('total_price', 0)),
                description=item.get('description', 'N/A')
            )
            invoice_details.items.append(invoice_item)

        # Add tax details
        for tax in details.get('tax_details', []):
            tax_detail = TaxDetails(
                document_id=document_id,
                tax_type=tax.get('tax_type', 'N/A'),
                tax_rate=float(tax.get('tax_rate', 0)),
                tax_amount=float(tax.get('tax_amount', 0)),
                description=tax.get('description', 'N/A')
            )
            invoice_details.tax_details.append(tax_detail)

        db.add(invoice_details)
        db.commit()
        return invoice_details
        
    except Exception as e:
        db.rollback()
        print(f"Error saving invoice details: {str(e)}")
        raise 