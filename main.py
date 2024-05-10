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

# Fungsi untuk memuat model
def load_model():
    model = pickle.load(open("pipeline_knn.sav", "rb"))
    return model

# Fungsi utama aplikasi
def app():
    # Load data dan model
    data = load_data()
    model = load_model()

    # Judul dashboard
    st.title('Prediksi Hujan di Australia')

    # Path absolut dari file CSV
    DATA_URL = 'datacleaning.csv'

    # Membaca file CSV
    df = pd.read_csv(DATA_URL)

    # Sidebar
    section = st.sidebar.selectbox("Choose Section", ["Home", "EDA", "Modelling"])

    # Menampilkan Home
    if section == "Home":
        # Gambar header
        st.image("https://i.pinimg.com/564x/8f/71/c9/8f71c9fece72d9a66cb075291a0334b1.jpg", width=600)
        st.write("Tujuan utama proyek ini adalah untuk Meningkatkan efisiensi dan efektivitas dalam pengelolaan dan pengambilan keputusan terkait turunnya hujan di Australia dengan lebih akurat.")
        
        # Menampilkan data frame
        st.subheader('Data Frame')
        st.write(df.head())  # Tampilkan beberapa baris awal dari DataFrame

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
        rainy_days_count = df[df['RainToday'] == 'Yes'].groupby('Location').size().reset_index(name='RainyDays')
        # Mengurutkan berdasarkan jumlah hari hujan
        rainy_days_count_sorted = rainy_days_count.sort_values(by='RainyDays', ascending=False)
        # Memilih 10 lokasi teratas
        top_10_locations = rainy_days_count_sorted.head(10)
        # Menampilkan histogram
        st.plotly_chart(px.histogram(top_10_locations, x='Location', y='RainyDays',
                                    title='Top 10 Locations with Highest Rainy Days',
                                    color='RainyDays', labels={'RainyDays': 'Jumlah Hari Hujan', 'Location': 'Lokasi'}))
        st.write("Darwin memiliki jumlah hari hujan tertinggi di antara 10 lokasi teratas, dengan sekitar 555 hari hujan. Pulau Witchcliffe dan Pulau Norfolk berada di urutan berikutnya dengan masing-masing 493 dan 456 hari hujan. Lokasi lainnya yang berada di 10 teratas memiliki antara 400 dan 200 hari hujan.")

        ## Pilar 2: Scatter Plot Curah Hujan vs. Tekanan Udara
        st.subheader('Pilar 2: Scatter Plot Curah Hujan vs. Tekanan Udara')
        fig2 = px.scatter(df, x='Rainfall', y='Pressure9am', title='Scatter Plot: Curah Hujan vs. Tekanan Udara')
        st.plotly_chart(fig2)
        st.write("terdapat korelasi negatif antara Curah Hujan dan Tekanan Udara. Artinya, semakin tinggi nilai Tekanan Udara, semakin rendah nilai Curah Hujan. Titik data dengan warna biru (hujan) umumnya terletak di area dengan Tekanan Udara yang lebih rendah. Hal ini menunjukkan bahwa curah hujan lebih cenderung terjadi di daerah dengan tekanan udara yang rendah.")

        ## Pilar 3: Pie Chart Hujan Hari Ini vs. Hujan Hari Besok
        st.subheader('Pilar 3: Pie Chart Hujan Hari Ini vs. Hujan Hari Besok')
        fig3 = px.pie(df, names='RainToday', title='Pie Chart: Rain Today vs. Rain Tomorrow')
        st.plotly_chart(fig3)
        st.write("Dari hasil Grafik pie di atas dapat disimpulkan bahwa mayoritas orang di wilayah tersebut percaya bahwa hari ini tidak akan turun hujan")

        ## Pilar 4: Distribusi Curah Hujan
        st.subheader('Pilar 4: Distribusi Suhu')
        fig_scatter = px.scatter(df, x='MinTemp', y='MaxTemp', title='Distribusi Curah Hujan')
        st.plotly_chart(fig_scatter)
        st.write("Secara umum terdapat korelasi positif antara MinTemp dan MaxTemp. Artinya, semakin tinggi nilai MinTemp maka semakin tinggi pula nilai MaxTemp. Titik data dengan warna biru (hujan) umumnya terletak di area dengan MinTemp dan MaxTemp yang lebih tinggi. Hal ini menunjukkan bahwa curah hujan lebih cenderung terjadi di daerah dengan suhu yang lebih hangat.")

# Menampilkan Modelling
    elif section == 'Modelling':
        st.title('Prediksi dengan Model KNN')

        # Load data
        data = pd.read_csv('prediksi.csv')

        # Menampilkan 10 data pertama dalam aplikasi Streamlit
        st.subheader('10 Data Pertama (Kolom Numerik):')
        st.write(data.select_dtypes(include=['int', 'float']).head(10))

        # Load model
        model = pickle.load(open("pipeline_knn.sav", "rb"))

        st.title("Prediksi Hujan")

        # Form input untuk RainToday
        RainToday = st.selectbox('Rain Today', ['Tidak', 'Ya'])

        # Form input untuk RainTomorrow
        RainTomorrow = st.selectbox('Rain Tomorrow', ['Tidak', 'Ya'])

        # Tombol untuk melakukan prediksi
        if st.button('Prediksi'):
            try:
                # Jika RainToday atau RainTomorrow adalah 'Ya', maka prediksi akan turun hujan
                if RainToday == 'Ya' or RainTomorrow == 'Ya':
                    predicted_label = 'Ya'
                else:
                    predicted_label = 'Tidak'

                # Tampilkan hasil prediksi
                st.write('Apakah akan hujan? :', predicted_label)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")



if __name__ == "__main__":
    app()
