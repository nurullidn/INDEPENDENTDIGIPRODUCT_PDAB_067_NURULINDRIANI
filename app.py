import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle

# Fungsi untuk memuat data
def load_data():
    data = pd.read_csv('datacleaning.csv')
    return data

# Fungsi utama aplikasi
def app():
    # Load data
    data = load_data()

    # Judul dashboard
    st.title('Prediksi Hujan di Australia')

    # Sidebar
    section = st.sidebar.selectbox("Choose Section", ["Home", "EDA", "Modelling"])

    # Menampilkan Home
    if section == "Home":
        # Gambar header
        st.image("https://i.pinimg.com/564x/8f/71/c9/8f71c9fece72d9a66cb075291a0334b1.jpg", width=600)
        st.write("Tujuan utama proyek ini adalah untuk Meningkatkan efisiensi dan efektivitas dalam pengelolaan dan pengambilan keputusan terkait turunnya hujan di Australia dengan lebih akurat.")
        
        # Menampilkan data frame
        st.subheader('Data Frame')
        st.write(data.head())  # Tampilkan beberapa baris awal dari DataFrame

        # Business Understanding
        st.subheader('Business Understanding')
        st.write("Proyek ini bertujuan untuk meningkatkan efisiensi dan efektivitas dalam pengelolaan dan pengambilan keputusan terkait turunnya hujan di Australia. Dengan memahami pola cuaca dan faktor-faktor yang memengaruhinya, diharapkan dapat memberikan prediksi yang lebih akurat tentang kemungkinan turunnya hujan di masa mendatang.")

    # Menampilkan EDA
    elif section == "EDA":
        # Visualisasi 4 pilar
        st.header('EDA')

        ## Pilar 1: Histogram Jumlah Hari Hujan per Lokasi
        st.subheader('Pilar 1: Histogram Jumlah Hari Hujan per Lokasi')
        # Menghitung jumlah hari hujan untuk setiap lokasi
        rainy_days_count = data[data['RainToday'] == 'Yes'].groupby('Location').size().reset_index(name='RainyDays')
        # Mengurutkan berdasarkan jumlah hari hujan
        rainy_days_count_sorted = rainy_days_count.sort_values(by='RainyDays', ascending=False)
        # Memilih 10 lokasi teratas
        top_10_locations = rainy_days_count_sorted.head(10)
        # Menampilkan histogram
        st.plotly_chart(px.histogram(top_10_locations, x='Location', y='RainyDays',
                                    title='Top 10 Locations with Highest Rainy Days',
                                    color='RainyDays', labels={'RainyDays': 'Jumlah Hari Hujan', 'Location': 'Lokasi'}))

        ## Pilar 2: Scatter Plot Curah Hujan vs. Tekanan Udara
        st.subheader('Pilar 2: Scatter Plot Curah Hujan vs. Tekanan Udara')
        fig2 = px.scatter(data, x='Rainfall', y='Pressure9am', title='Scatter Plot: Curah Hujan vs. Tekanan Udara')
        st.plotly_chart(fig2)

        ## Pilar 3: Pie Chart Hujan Hari Ini vs. Hujan Hari Besok
        st.subheader('Pilar 3: Pie Chart Hujan Hari Ini vs. Hujan Hari Besok')
        fig3 = px.pie(data, names='RainToday', title='Pie Chart: Rain Today vs. Rain Tomorrow')
        st.plotly_chart(fig3)

        ## Pilar 4: Distribusi Curah Hujan
        st.subheader('Pilar 4: Distribusi Curah Hujan')
        fig4 = px.histogram(data, x='Rainfall', title='Distribution of Rainfall')
        st.plotly_chart(fig4)

    # Menampilkan Modelling
    # PREDIKSI
    elif section == 'Modelling':
        st.title('Prediksi dengan Model KNN')
        
        # Menampilkan 10 data pertama dari kolom dengan tipe data numerik saja

        # Load data
        data = pd.read_csv('dataclean.csv')

        # Menampilkan 10 data pertama dalam aplikasi Streamlit
        st.subheader('10 Data Pertama (Kolom Numerik):')
        st.write(data.select_dtypes(include=['int', 'float']).head(10))

        # Load model
        model = joblib.load(open("knn.sav", "rb"))

        st.title("Prediksi Hujan")

                # Get inputs
        min_temp = st.number_input('Minimum Temperature', value=0.0, step=0.1)
        max_temp = st.number_input('Maximum Temperature', value=0.0, step=0.1)
        rainfall = st.number_input('Rainfall', value=0.0, step=0.1)
        evaporation = st.number_input('Evaporation', value=0.0, step=0.1)
        sunshine = st.number_input('Sunshine', value=0.0, step=0.1)
        wind_gust_speed = st.number_input('Wind Gust Speed', value=0.0, step=0.1)
        wind_speed_9am = st.number_input('Wind Speed 9am', value=0.0, step=0.1)
        wind_speed_3pm = st.number_input('Wind Speed 3pm', value=0.0, step=0.1)
        humidity_9am = st.number_input('Humidity 9am', value=0.0, step=0.1)
        humidity_3pm = st.number_input('Humidity 3pm', value=0.0, step=0.1)
        pressure_9am = st.number_input('Pressure 9am', value=0.0, step=0.1)
        pressure_3pm = st.number_input('Pressure 3pm', value=0.0, step=0.1)
        cloud_9am = st.number_input('Cloud 9am', value=0.0, step=0.1)
        cloud_3pm = st.number_input('Cloud 3pm', value=0.0, step=0.1)
        temp_9am = st.number_input('Temperature 9am', value=0.0, step=0.1)
        temp_3pm = st.number_input('Temperature 3pm', value=0.0, step=0.1)
        rain_today = st.selectbox('Rain Today', ['No', 'Yes'])
        rain_tomorrow = st.selectbox('Rain Tomorrow', ['No', 'Yes'])

        # Button to trigger prediction
        if st.button('Predict'):
            # Dynamic text
            prediction_state = st.markdown('Calculating...')

            # Create DataFrame for the weather data
            weather_data = pd.DataFrame({
                'MinTemp': [min_temp],
                'MaxTemp': [max_temp],
                'Rainfall': [rainfall],
                'Evaporation': [evaporation],
                'Sunshine': [sunshine],
                'WindGustSpeed': [wind_gust_speed],
                'WindSpeed9am': [wind_speed_9am],
                'WindSpeed3pm': [wind_speed_3pm],
                'Humidity9am': [humidity_9am],
                'Humidity3pm': [humidity_3pm],
                'Pressure9am': [pressure_9am],
                'Pressure3pm': [pressure_3pm],
                'Cloud9am': [cloud_9am],
                'Cloud3pm': [cloud_3pm],
                'Temp9am': [temp_9am],
                'Temp3pm': [temp_3pm],
                'RainToday': [rain_today],
                'RainTomorrow': [rain_tomorrow]  # Tambahkan koma di sini
            })


             # Make prediction
            y_pred = model.predict(weather_data)

            # Determine the predicted label
            if rain_today == 'Yes' or rain_tomorrow == 'Yes':
                predicted_label = 'Yes'
            else:
                predicted_label = 'No'

            # Display prediction result
            if y_pred[0] == 0:
                msg = f'This weather is predicted to: **not experience rain tomorrow** : {predicted_label}'
            else:
                msg = f'This weather is predicted to: **experience rain tomorrow** : {predicted_label}'

            prediction_state.markdown(msg)

if __name__ == "__main__":
    app()
