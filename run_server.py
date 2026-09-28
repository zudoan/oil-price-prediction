# run_server.py
"""
PetroForecast AI — Local Server Launcher
Chạy máy chủ web FastAPI tại: http://localhost:8000
Tự động phát hiện và chuyển sang Python 3.11 (RTX 4060 Ti GPU) nếu chạy bằng Python khác.
"""
import sys
import os
import subprocess
import socket

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Auto-switch to Python 3.11 if current interpreter lacks keras or torch
try:
    import keras
    import torch
except ImportError:
    py311 = r"C:\Users\AD\AppData\Local\Programs\Python\Python311\python.exe"
    if os.path.exists(py311) and sys.executable.lower() != py311.lower():
        print("⚡ Phát hiện Python mặc định chưa có Keras/PyTorch.")
        print(f"🔄 Đang tự động chuyển sang Python 3.11 GPU ({py311})...\n")
        cmd = [py311, "-X", "utf8", os.path.abspath(__file__)] + sys.argv[1:]
        sys.exit(subprocess.call(cmd))

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.backend.config import HOST, PORT, APP_NAME, VERSION

def is_port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║               {APP_NAME:^54s} ║
║               Phiên bản: {VERSION:^43s} ║
╠══════════════════════════════════════════════════════════════════════╣
║  • Máy chủ Web:     http://localhost:{PORT:<5d}                          ║
║  • Tài liệu API:    http://localhost:{PORT:<5d}/docs                     ║
║  • Giao diện UI:    http://localhost:{PORT:<5d} (Dashboard Analytics)    ║
║  • Công nghệ:       FastAPI · Keras 3 · PyTorch CUDA · ApexCharts   ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    if is_port_busy(PORT):
        print(f"🟢 MÁY CHỦ PETROFORECAST AI ĐANG CHẠY SẴN TRÊN CỔNG {PORT}!")
        print(f"👉 Bạn chỉ cần mở trình duyệt và truy cập: http://localhost:{PORT}")
        print(f"👉 Tài liệu API Swagger:               http://localhost:{PORT}/docs\n")
        sys.exit(0)

    import uvicorn
    uvicorn.run("app.backend.main:app", host="127.0.0.1", port=PORT, reload=False)
