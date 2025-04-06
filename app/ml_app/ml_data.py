import httpx
import asyncio
import datetime

'''
async def ml_listener(trip_id):
    kilometer = 1
    post = 0
    try:
        print(f"Listening for ML events for trip {trip_id}...")
        async with httpx.AsyncClient() as client:
            while True:
                await asyncio.sleep(3.5)
                post += 1
                kilometer += post // 10
                post %= 10
                input_data = {
                    'trip_id': trip_id,
                    'kilometer_value': kilometer,
                    'picket_value': post,
                    'time_track': datetime.datetime.now().isoformat()
                }
                print('_____________')
                print(f'ML DATA: {input_data}')
                print('_____________')
                # Use the async client to send the post request
                response = await client.post('http://localhost:8000/api/trip/put_ml', json=input_data)
                print(f"Response Status: {response.status_code}")
    except asyncio.CancelledError:
        print(f"ML listener for trip {trip_id} is stopping.")
        '''

from ultralytics import YOLOWorld

async def ml_listener(trip_id):
    model = YOLOWorld("yolo8x-worldv2.pt")
    model.set_classes(['picket kilometer post'])
    try:
        print(f"Listening for ML events for trip {trip_id}...")
        async with httpx.AsyncClient() as client:
            while True:
                await asyncio.sleep(3.5)
                
                
                input_data = {
                    'trip_id': trip_id,
                    'kilometer_value': kilometer,
                    'picket_value': post,
                    'time_track': datetime.datetime.now().isoformat()
                }
                response = await client.post('http://localhost:8000/api/trip/put_ml', json=input_data)
                print(f"Response Status: {response.status_code}")
    except asyncio.CancelledError:
        print(f"ML listener for trip {trip_id} is stopping.")