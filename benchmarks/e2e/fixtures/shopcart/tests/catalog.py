"""Shared synthetic catalog for shopcart tests. All products are fictional."""

from shopcart.models import Product

BOOK = Product("BOOK-001", "Intro to Testing", 1995, weight_grams=450)
NOTEBOOK = Product("BOOK-002", "Blank Notebook", 499, weight_grams=200)
MUG = Product("MUG-100", "Lab Mug", 825, weight_grams=350)
TSHIRT = Product("TEE-L", "Logo T-Shirt L", 1500, weight_grams=180)
