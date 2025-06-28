from locust import HttpUser, task, between
import numpy as np
import cv2


class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        self.frame = buf.tobytes()

    @task
    def predict(self):
        files = {'file': ('frame.jpg', self.frame, 'image/jpeg')}
        self.client.post('/predict', files=files)
