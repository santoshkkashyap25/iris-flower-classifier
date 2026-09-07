import unittest
from app import create_app

class TestAppRoutes(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Iris Flower Predictor', response.data)
        self.assertIn(b'Classify Species', response.data)

    def test_get_predict_redirects(self):
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith('/') or response.location == '/')

    def test_post_predict_semantic_keys(self):
        form_data = {
            'sepal_length': '5.1',
            'sepal_width': '3.5',
            'petal_length': '1.4',
            'petal_width': '0.2'
        }
        response = self.client.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Iris Setosa', response.data)
        self.assertIn(b'Match', response.data)

    def test_post_predict_legacy_keys(self):
        form_data = {
            'a': '5.1',
            'b': '3.5',
            'c': '1.4',
            'd': '0.2'
        }
        response = self.client.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Iris Setosa', response.data)

    def test_post_predict_out_of_range_flash_warning_does_not_crash(self):
        # Sepal length 99.0 is out of bounds, should flash warning without 500 session error
        form_data = {
            'sepal_length': '99.0',
            'sepal_width': '3.5',
            'petal_length': '1.4',
            'petal_width': '0.2'
        }
        response = self.client.post('/predict', data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # Should not crash and should render after page or home with warning
        self.assertIn(b'Prediction:', response.data)

    def test_post_predict_invalid_input(self):
        form_data = {
            'sepal_length': 'not-a-number',
            'sepal_width': '3.5',
            'petal_length': '1.4',
            'petal_width': '0.2'
        }
        response = self.client.post('/predict', data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid input', response.data)

    def test_api_predict_success(self):
        payload = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['prediction']['species'], 'Iris-setosa')
        self.assertIn('probabilities', data)

    def test_api_predict_list_features(self):
        payload = {
            'features': [6.5, 3.0, 5.5, 2.0]
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['prediction']['species'], 'Iris-virginica')

    def test_api_predict_non_json(self):
        response = self.client.post('/api/predict', data="not json")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data['success'])

    def test_404_error_page(self):
        response = self.client.get('/this-route-does-not-exist')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'404', response.data)
        self.assertIn(b'Specimen Page Not Found', response.data)

if __name__ == '__main__':
    unittest.main()
