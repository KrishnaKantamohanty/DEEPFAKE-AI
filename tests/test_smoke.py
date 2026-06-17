import unittest
from pathlib import Path

import torch

from app import app
from model import get_model
from predict import preprocess_image


class ProjectSmokeTests(unittest.TestCase):
    def test_model_forward_shape(self):
        model = get_model(fine_tune=False, pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (1, 1))

    def test_preprocess_known_image(self):
        image_path = Path('data/real/1.jpg')
        self.assertTrue(image_path.exists(), 'Expected sample image data/real/1.jpg to exist')
        tensor = preprocess_image(str(image_path))
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))

    def test_flask_health_and_home(self):
        client = app.test_client()
        health = client.get('/health')
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()['status'], 'ok')

        home = client.get('/')
        self.assertEqual(home.status_code, 200)
        self.assertIn(b'Fake Image Detector', home.data)


if __name__ == '__main__':
    unittest.main()
