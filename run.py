"""
multi-agent-fits-dev-02/run.py
"""

import uvicorn
import sys
from pathlib import Path

# Add the parent directory to the sys.path to ensure modules can be found
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,     # Auto-reload on code changes
        log_level="info"
    )