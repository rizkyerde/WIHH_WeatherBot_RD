# Weather Polymarket - Jakarta Tmax Pipeline

Pipeline untuk ingest METAR data WIHH (Jakarta Halim) dari Ogimet,
parsing ke SQLite database untuk modeling Tmax harian.

## Setup

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt