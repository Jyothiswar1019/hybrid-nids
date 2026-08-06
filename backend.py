import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS so Streamlit Cloud can access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
combined_app = socketio.ASGIApp(sio, app)

# Global variable to hold the latest sniffed features
latest_features = None

@sio.event
async def connect(sid, environ):
    print(f"Sniffer Agent connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Sniffer Agent disconnected: {sid}")

@sio.on('live_features')
async def handle_features(sid, data):
    global latest_features
    print(f"Received packet features: {data}")
    latest_features = data

@app.get("/get_features")
async def get_features():
    global latest_features
    # Return the data and immediately clear it so we don't process the same packet twice
    data_to_return = latest_features
    latest_features = None
    return data_to_return

if __name__ == "__main__":
    uvicorn.run(combined_app, host="0.0.0.0", port=8000)