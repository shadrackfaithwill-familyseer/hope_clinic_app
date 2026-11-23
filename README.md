# Hope Health Clinic AI Assistant

## Setup

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env`.
5. Add your Gemini API key to `.env`.
6. Run:

   ```bash
   python main.py
   ```

## Features

- Gemini API integration
- English, Swahili and Sheng clinic FAQ support
- Full in-session conversation history sent on every answer request
- JSON response parsing and validation, with one retry
- Final AI-generated session summary
- Dated JSON session files
- Menu to start a session, open a past session, or exit

## Important limitation

This project provides booking guidance but is not connected to a live appointment
database. It must not tell a patient that an appointment is confirmed.

## Privacy

Session files may contain information typed by the user. Use fictional details
for classroom demonstrations, keep `.env` private, and do not commit real patient
conversation files to a public GitHub repository.
