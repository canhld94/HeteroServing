# Note on the API server

## API specificiation

### Request format

### Response format

Example response of `POST /inference`

```json
{
    "status": "ok",
    "predictions": [
        {
            "label_id": "78",
            "label": "lalaland_0",
            "confidences": "1.1627906976744187",
            "detection_box": [
                "777",
                "915",
                "793",
                "335"
            ]
        },
        {
            "label_id": "38",
            "label": "lalaland_1",
            "confidences": "1.0869565217391304",
            "detection_box": [
                "649",
                "421",
                "362",
                "27"
            ]
        },
        {
            "label_id": "13",
            "label": "lalaland_2",
            "confidences": "1.6949152542372881",
            "detection_box": [
                "763",
                "926",
                "540",
                "426"
            ]
        }
    ]
}
```
