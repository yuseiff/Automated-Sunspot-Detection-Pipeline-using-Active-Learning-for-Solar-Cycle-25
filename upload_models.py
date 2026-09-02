import os  # <--- 1. Import os
from roboflow import Roboflow
from dotenv import load_dotenv

# 2. Load the .env file (returns True/False, but we don't need to capture it)
load_dotenv() 

# 3. Fetch the actual key string using os.getenv
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

# (Optional) Verify the key was found
if not roboflow_api_key:
    print("Error: ROBOFLOW_API_KEY not found in environment variables.")
else:
    rf = Roboflow(api_key=roboflow_api_key)
    workspace = rf.workspace("research-8avrk")

    workspace.deploy_model(
      model_type="rfdetr-base",
      model_path=r"Computer Vision Code\RF-DETR\RF-DETR_S_V3_V6",
      project_ids=["sunspots-n3q0k"],
      model_name="RF-DETR-S-V3-V6"
    )