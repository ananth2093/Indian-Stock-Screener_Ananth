# Raspberry Pi deployment

This folder contains systemd service/timer files to keep the Indian screener running locally on a Pi.

## Setup

1. Copy the repo to the Pi:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Indian-Stock-Screener_Ananth.git /home/pi/indian-screener
   cd /home/pi/indian-screener
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Test the updater once:
   ```bash
   python update_data.py
   ```

4. Install the systemd services:
   ```bash
   chmod +x deploy/pi-install.sh
   ./deploy/pi-install.sh
   ```

## What the services do

- `pi-screener-dashboard.service` — runs `streamlit run app.py` on port 8501, starts on boot, and restarts if it crashes.
- `pi-screener-updater.service` — runs `python update_data.py` when triggered.
- `pi-screener-updater.timer` — triggers the updater every hour.

`update_data.py` has a built-in throttle:
- It runs every hour while the Indian market is open (roughly 09:15–15:30 IST, weekdays).
- Outside market hours it runs at most every 4 hours.
- If no snapshot exists yet, it always runs.

## Access the dashboard

From any device on your home network:

```text
http://YOUR_PI_IP:8501
```

## Useful commands

```bash
# View dashboard status
sudo systemctl status pi-screener-dashboard

# View updater timer status
sudo systemctl status pi-screener-updater.timer

# View latest updater run
sudo journalctl -u pi-screener-updater -n 100 -f

# Run updater manually
sudo systemctl start pi-screener-updater

# Stop dashboard
sudo systemctl stop pi-screener-dashboard
```
