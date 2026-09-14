from inference_sdk import InferenceHTTPClient, InferenceConfiguration

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="Se0BlQ1umhw7BwKExdDg"
).configure(InferenceConfiguration(
#    api_key_transport="header"  # header-based auth (inference v1.5.0+)
))

result = CLIENT.infer(r"Raw Data\2025\07\31\20250731_060000_Ic_flat_4k.jpg", model_id="sunspots-n3q0k/70")

# save the results image
with open(r"runs\detect\predict\20250731_060000_Ic_flat_4k_result.jpg", "wb") as f:
    f.write(result.image)