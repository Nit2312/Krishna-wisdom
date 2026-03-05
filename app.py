"""Flask entrypoint for Vercel.

This file intentionally exports the Flask `app` from api/app.py to avoid
Streamlit conflicts and ensure Vercel detects the Flask application.
"""

from api.app import app


if __name__ == "__main__":
    app.run()

