"""Dùng backend không giao diện cho kiểm thử hình trên Windows và CI."""
import os

os.environ.setdefault('MPLBACKEND', 'Agg')
