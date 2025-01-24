import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
st.set_page_config(page_title='پیش بینی وقوع سیل - RoboAi', layout='centered', page_icon='☂️')

def load_model():
    with open('saved.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
x = data['x']

def show_page():
    st.write("<h4 style='text-align: center; color: blue;'>پیش بینی سیل خیز بودن منطقه 🌧️</h4>", unsafe_allow_html=True)
    st.write("<h6 style='text-align: center; color: black;'>Robo-Ai.ir طراحی و توسعه</h6>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")
    container = st.container(border=True)
    container.write("<h6 style='text-align: right; color: gray;'>پیش بینی سیل خیز بودن هر منطقه بر اساس توپوگرافی با هوش مصنوعی ☂️</h6>", unsafe_allow_html=True)
    st.write('')

    with st.sidebar:
        st.write("<h5 style='text-align: center; color: blcak;'>پیش بینی وقوع سیل بر اساس آمار توپوگرافی منطقه مورد نظر</h5>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: center; color: gray;'>ساخته شده برای تشخیص مناطق مناسب ساخت و ساز شهری و صنعتی</h5>", unsafe_allow_html=True)
        st.divider()
        st.write("<h5 style='text-align: center; color: black;'>طراحی و توسعه</h5>", unsafe_allow_html=True)
        st.write("<h5 style='text-align: center; color: gray;'>حمیدرضا بهرامی</h5>", unsafe_allow_html=True)

    MonsoonIntensity = st.slider('شدت باد های موسمی منطقه چقدر است؟', 0.0, 16.0, 1.0)
    st.divider()

    TopographyDrainage = st.slider('به توپوگرافی زهکشی منطقه چه نمره ای می دهید؟', 0.0, 18.0, 1.0)
    st.divider()

    RiverManagement = st.slider('به مدیریت رودخانه های منطقه چه نمره ای می دهید؟', 0.0, 16.0, 1.0)
    st.divider()

    Deforestation = st.slider('چه میزان از خاک منطقه جنگل زدایی شده است؟', 0.0, 17.0, 1.0)
    st.divider()

    Urbanization = st.slider('شهرنشینی در منطقه چقدر است؟', 0.0, 17.0, 1.0)
    st.divider()

    ClimateChange = st.slider('به تغییرات اقلیمی منطقه در طول سال های اخیر چه نمره ای می دهید؟', 0.0, 17.0, 1.0)
    st.divider()

    DamsQuality = st.slider('سد های منطقه از نظر کیفیت و مقاومت چه نمره ای می گیرند؟', 0.0, 16.0, 1.0)
    st.divider()

    Siltation = st.slider('غلظت رسوبات معلق در رودخانه های منطقه چقدر است؟', 0.0, 16.0, 1.0)
    st.divider()

    AgriculturalPractices = st.slider('کیفیت کشاورزی اصولی در منطقه چطور است؟', 0.0, 16.0, 1.0)
    st.divider()

    Encroachments = st.slider('چه میزان از زیرساخت های صنعتی و شهری در مسیر سیلاب توسعه یافته است؟', 0.0, 18.0, 1.0)
    st.divider()

    DrainageSystems = st.slider('به سیستم های زهکشی منطقه چه نمره ای می دهید؟', 0.0, 17.0, 1.0)
    st.divider()

    CoastalVulnerability = st.slider('آسیب پذیری ساحلی منطقه چقدر است؟', 0.0, 17.0, 1.0)
    st.divider()

    Landslides = st.slider('احتمال رانش زمین در منطقه چقدر است؟', 0.0, 16.0, 1.0)
    st.divider()

    Watersheds = st.slider('حوزه های آبخیز منطقه چقدر است؟', 0.0, 16.0, 1.0)
    st.divider()

    DeterioratingInfrastructure = st.slider('چه میزان از زیرساخت های منطقه در حال پوسیدگی و از بین رفتن است؟', 0.0, 17.0, 1.0)
    st.divider()

    PopulationScore = st.slider('امتیاز جمعیتی منطقه چقدر است؟', 0.0, 19.0, 1.0)
    st.divider()

    WetlandLoss = st.slider('هر سال چه میزان از تالاب های منطقه از بین می رود؟', 0.0, 22.0, 1.0)
    st.divider()

    InadequatePlanning = st.slider('فکر می کنید برنامه ریزی ناکافی چقدردر بروز مشکلات منطقه موردنظر موثر است؟', 0.0, 16.0, 1.0)
    st.divider()

    PoliticalFactors = st.slider('آیا ایجاد تغییر در منطقه مورد نظر تحت تاثیر تصمیمات سیاسی است؟', 0.0, 16.0, 1.0)
    st.divider()

    IneffectiveDisasterPreparedness = st.slider('در صورت وقوع سیل ، آمادگی نیرو های امداد و زیرساخت را چطور ارزیابی می کنید؟', 0.0, 16.0, 1.0)
    st.divider()

    button = st.button('ارزیابی')
    if button:
        with st.chat_message("assistant"):
                with st.spinner('''درحال ارزیابی ، لطفا صبور باشید'''):
                    time.sleep(2)
                    st.success(u'\u2713''ارزیابی انجام شد')
                    x = np.array([[MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation,
                                    Urbanization, ClimateChange, DamsQuality, Siltation, AgriculturalPractices,
                                    Encroachments, DrainageSystems, CoastalVulnerability, Landslides, Watersheds, DeterioratingInfrastructure,
                                    PopulationScore, WetlandLoss, InadequatePlanning, PoliticalFactors, IneffectiveDisasterPreparedness]])

        y_prediction = model.predict(x)
        if y_prediction >= 0.50:
            text1 = 'بر اساس ارزیابی من ، احتمال وقوع سیل در منطقه موردنظر بالاست'
            text2 = 'در منطقه مورد نظر ساخت و ساز نکنید'
            text3 = 'Based on my analysis, This region is Highly vulnerable to flood'
            text4 = "It's wiser to choose another region for your constructions"
            def stream_data1():
                for word in text1.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data1)
            def stream_data2():
                for word in text2.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data2)
            def stream_data3():
                for word in text3.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data3)
            def stream_data4():
                for word in text4.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data4)
            
        else:
            text5 = 'بر اساس ارزیابی من ، احتمال وقوع سیل در منطقه موردنظر کم است'
            text6 = 'می توانید در منطقه مورد نظر ساخت و ساز کنید'
            text7 = 'Based on my analysis, This region is Not vulnerable to flood'
            text8 = "You can choose this region for your constructions"
            def stream_data5():
                for word in text5.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data5)
            def stream_data6():
                for word in text6.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data6)
            def stream_data7():
                for word in text7.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data7)
            def stream_data8():
                for word in text8.split(" "):
                    yield word + " "
                    time.sleep(0.09)
            st.write_stream(stream_data8)
show_page()
