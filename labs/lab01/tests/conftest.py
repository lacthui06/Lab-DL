"""Dung backend khong giao dien cho kiem thu hinh tren Windows va CI."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
