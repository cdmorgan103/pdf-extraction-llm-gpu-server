"""
schemas.py - the table formats the LLM must fill in.

Each class below describes one kind of document: which fields it has, what type
each field is, and what each field means. Every other script relies on these
definitions, so this file is the single place to change if you want to extract
different fields or add a new document type.

extract.py uses these classes in three ways:

  1. Constrained output. The class is turned into a JSON Schema and passed to
     Ollama. Ollama then only lets the model produce JSON with exactly these
     fields and types, so the reply can always be parsed.

  2. Instructions. The same JSON Schema, including each field's description,
     is pasted into the prompt so the model knows what each field means.

  3. Validation. The model's reply is checked against the class (Pydantic
     converts types and rejects anything that does not fit) before it is saved.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Invoices (synthetic dataset)
# An invoice becomes two tables: one row per invoice (the header fields below)
# and one row per line item (the LineItem class).
# ---------------------------------------------------------------------------

# Note: these classes deliberately have no docstrings. Pydantic copies a class
# docstring into the JSON Schema, which is sent to the model, so adding one would
# silently change the prompt. Use comments (like this one) instead.

# One row of the invoice's item table.
class LineItem(BaseModel):
    line_no: int = Field(description="1-based position of the line in the item table")
    description: str = Field(description="item or service description, as printed")
    quantity: float
    unit_price: float = Field(description="price per unit, plain number")
    amount: float = Field(description="line total, plain number")


# Everything we want from one invoice.
class InvoiceExtraction(BaseModel):
    # Identification and dates
    invoice_number: str = Field(description="the invoice's own number / reference, exactly as printed")
    invoice_date: str = Field(description="issue date, ISO format YYYY-MM-DD")
    due_date: Optional[str] = Field(description="payment due date, ISO format YYYY-MM-DD, null if absent")

    # The two parties
    vendor_name: str = Field(description="company issuing the invoice (the seller)")
    vendor_address: str = Field(description="seller address on one line, parts separated by ', '")
    customer_name: str = Field(description="who is billed (the buyer)")
    customer_address: str = Field(description="buyer address on one line, parts separated by ', '")

    # Money
    currency: str = Field(description="ISO 4217 code, e.g. USD, GBP, EUR")
    subtotal: float = Field(description="total before tax and shipping")
    tax_rate: float = Field(description="tax rate as a fraction, e.g. 0.2 for 20%")
    tax_amount: float
    shipping: float = Field(description="shipping / delivery / freight charge, 0 if none")
    total: float = Field(description="final amount due")

    # The item table
    line_items: list[LineItem]


# ---------------------------------------------------------------------------
# Receipts (SROIE dataset)
# These four fields are the ones the SROIE benchmark labels.
# ---------------------------------------------------------------------------

# Everything we want from one receipt.
class ReceiptExtraction(BaseModel):
    company: str = Field(description="name of the store / company that issued the receipt, as printed")
    address: str = Field(description="store address as printed, on one line")
    date: str = Field(description="transaction date, ISO format YYYY-MM-DD")
    total: float = Field(description="final total paid, including tax and rounding")


# Lookup table used by the command-line option --schema (e.g. --schema invoice).
SCHEMAS = {
    "invoice": InvoiceExtraction,
    "receipt": ReceiptExtraction,
}
