import torch
import streamlit as st
from PIL import Image
from diffusers import DiffusionPipeline as DP

# สร้าง header และ subheader
h = st.header('Diffusion.AI')
s = st.subheader('เว็บไซต์สำหรับแปลงข้อความเป็นภาพ')

# แสดงข้อความ
p = st.write('เว็บไซต์นี้แลกมาด้วยหยาดเหงื่อและความอดทน')

# รับข้อความจากผู้ใช้
text = st.text_input('prompt: ')

# ตรวจสอบว่ามีข้อความหรือไม่
if text:
    # สร้าง DiffusionPipeline
    dp = DP.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    
    # สร้างรูปภาพจากข้อความ
    image_data = dp(text).images[0]
    image = Image.fromarray(image_data)
    
    # แสดงรูปภาพใน Streamlit UI
    st.image(image, caption='Generated Image', use_column_width=True)

    # แสดงปุ่ม
    b = st.button('จะไปต่อหรือ.......')

    # ตรวจสอบการคลิกปุ่ม
    if b:
        st.success('คุณได้กดปุ่ม "จะไปต่อ" แล้ว')
