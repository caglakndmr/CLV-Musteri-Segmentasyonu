import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable,q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def cltv_data_preb(df):

    df.columns = [col.upper() for col in df.columns]

    # Müşteri yakın zamanda (5 ay öncesine kadar) claimde bulunmuş mu?
    df.loc[df["MONTHS SINCE LAST CLAIM"] <= 5, "NEW_Recent Claim"] = "Yes"
    df.loc[df["MONTHS SINCE LAST CLAIM"] > 5, "NEW_Recent Claim"] = "No"
    # Ortalama poliçe fiyatı
    df["NEW_Average Policy Price"] = df["MONTHLY PREMIUM AUTO"] / df["NUMBER OF POLICIES"]
    # Açık şikayet var mı?
    df.loc[df["NUMBER OF OPEN COMPLAINTS"] == 0, "OPEN COMPLAINTS"] = "No"
    df.loc[df["NUMBER OF OPEN COMPLAINTS"] > 0, "OPEN COMPLAINTS"] = "Yes"
    # Son şikayet ne zaman oldu?
    df.loc[df["MONTHS SINCE LAST CLAIM"] <= 10, "NEW_LAST_CLAİM"] = "Recently"
    df.loc[(df["MONTHS SINCE LAST CLAIM"] > 10) & (df["MONTHS SINCE LAST CLAIM"] <= 20), "NEW_LAST_CLAİM"] = "Medium"
    df.loc[df["MONTHS SINCE LAST CLAIM"] > 20, "NEW_LAST_CLAİM"] = "Long"
    # Aylık sigorta ödemesi durumu
    df.loc[df["MONTHLY PREMIUM AUTO"] <= 70, "NEW_PAYMENT_STATUS"] = "Underpayment"
    df.loc[(df["MONTHLY PREMIUM AUTO"] > 70) & (df["MONTHLY PREMIUM AUTO"] < 100), "NEW_PAYMENT_STATUS"] = "Medium"
    df.loc[df["MONTHLY PREMIUM AUTO"] >= 100, "NEW_PAYMENT_STATUS"] = "Overpayment"
    # Gelir durumu
    df.loc[df["INCOME"] == 0, "NEW_INCOME_STATUS"] = "No Income"
    df.loc[(df["INCOME"] > 0) & (df["INCOME"] <= 35000), "NEW_INCOME_STATUS"] = "Low Income"
    df.loc[(df["INCOME"] > 35000) & (df["INCOME"] <= 60000), "NEW_INCOME_STATUS"] = "Medium Income"
    df.loc[df["INCOME"] > 60000, "NEW_INCOME_STATUS"] = "High Income"
    # Müşteriye ödenen sigorta parası
    df.loc[df["TOTAL CLAIM AMOUNT"] <= 200, "NEW_CLAIM_STATUS"] = "Low"
    df.loc[(df["TOTAL CLAIM AMOUNT"] > 200) & (df["TOTAL CLAIM AMOUNT"] <= 550), "NEW_CLAIM_STATUS"] = "Medium"
    df.loc[df["TOTAL CLAIM AMOUNT"] > 550, "NEW_CLAIM_STATUS"] = "High"
    # Müşteriden elde edilen kar zarar miktarı
    df["NEW_PROFIT"] = (df["MONTHLY PREMIUM AUTO"] * df["MONTHS SINCE POLICY INCEPTION"])
    df.loc[df["NUMBER OF OPEN COMPLAINTS"] > 0, "NEW_PROFIT"] = (df["MONTHLY PREMIUM AUTO"] * df["MONTHS SINCE POLICY INCEPTION"]) - df["TOTAL CLAIM AMOUNT"]
    # kazanç-kayıp durumu
    df.loc[df["NEW_PROFIT"] <= 0, "NEW_PROFIT-LOSS"] = "Loss"
    df.loc[df["NEW_PROFIT"] > 0, "NEW_PROFIT-LOSS"] = "Gain"
    # Müşterinin ödediği aylık sigorta parasının ortalama ödenen sigorta parasına oranı
    df["NEW_PAYMENT RATE"] = df["MONTHLY PREMIUM AUTO"] / df["MONTHLY PREMIUM AUTO"].mean()
    # Müşterilerin yıllık gelirlerin ortalama yıllık gelire oranı
    df["NEW_INCOME RATE"] = df["INCOME"] / df["INCOME"].mean()
    # Since Claim - Monthly premıum auto ilişkisi
    df.loc[df["MONTHS SINCE LAST CLAIM"] > 25, "NEW_CLAIM-INCOME"] = df.loc[df["MONTHS SINCE LAST CLAIM"] > 25, "MONTHLY PREMIUM AUTO"] / df.loc[df["MONTHS SINCE LAST CLAIM"] > 25, "MONTHLY PREMIUM AUTO"].mean()
    df.loc[(df["MONTHS SINCE LAST CLAIM"] > 10) & (df["MONTHS SINCE LAST CLAIM"] <= 25), "NEW_CLAIM-INCOME"] = df.loc[(df["MONTHS SINCE LAST CLAIM"] > 10) & (df["MONTHS SINCE LAST CLAIM"] <= 25), "MONTHLY PREMIUM AUTO"] / df.loc[(df["MONTHS SINCE LAST CLAIM"] > 10) & (df["MONTHS SINCE LAST CLAIM"] <= 25), "MONTHLY PREMIUM AUTO"].mean()
    df.loc[df["MONTHS SINCE LAST CLAIM"] <= 10, "NEW_CLAIM-INCOME"] = df.loc[df["MONTHS SINCE LAST CLAIM"] <= 10, "MONTHLY PREMIUM AUTO"] / df.loc[df["MONTHS SINCE LAST CLAIM"] <= 10, "MONTHLY PREMIUM AUTO"].mean()
    # Ödenen Sigorta Parası Ortalamadan Fazla mı?
    df.loc[df["TOTAL CLAIM AMOUNT"] > df["TOTAL CLAIM AMOUNT"].mean(), "NEW_CLAIM_CAT"] = "High"
    df.loc[df["TOTAL CLAIM AMOUNT"] < df["TOTAL CLAIM AMOUNT"].mean(), "NEW_CLAIM_CAT"] = "Low"
    # Sigorta Geri Ödemesi Bir Önceki Sözleşmeye mi ait?
    df["NEW_PREVIOUS AGREE"] = "No"
    df.loc[df["MONTHS SINCE LAST CLAIM"] > df["MONTHS SINCE POLICY INCEPTION"], "NEW_PREVIOUS AGREE"] = "Yes"
    # Poliçe sayısı 2'den küçük mü?
    df.loc[df["NUMBER OF POLICIES"] <= 2, "NEW_FEW POLİCY"] = "Yes"
    df.loc[df["NUMBER OF POLICIES"] > 2, "NEW_FEW POLİCY"] = "No"

    df.columns = [col.upper() for col in df.columns]

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

    df = one_hot_encoder(df, cat_cols, drop_first=True)

    df.columns = [col.upper() for col in df.columns]


    X_scaled = MinMaxScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    df = df.drop("CUSTOMER LIFETIME VALUE", axis=1)

    features = pd.DataFrame(df.iloc[-1]).T

    return features


st.set_page_config(layout="wide")

st.image("TURKCELL_YATAY_ERKEK_LOGO.png", width=150)
st.image("logo-dark.png", width=150)
st.sidebar.image("indir-removebg-preview.png", width=100)

st.title("CLTV Tahmini ve Müşteri Segmentasyonu")
st.title("")

################# Read Datasets #################
df = pd.read_csv('squark_automotive_CLV_training_data.csv')
df.dropna(inplace=True)
production = pd.read_csv("squark_automotive_CLV_production_data.csv")
segments = pd.read_csv("New_Segments.csv")
segments.drop("Unnamed: 0", axis=1, inplace=True)
df = df.drop(["Effective To Date", "Customer"], axis=1)
production = production.drop(["Effective To Date", "Customer"], axis=1)
###################################################

# MAIN PAGE

# Show Dataset and Choosing Customer
customer = st.text_input("Müşteri ID'si:")
if customer != "":
    st.write(segments[segments["Customer"] == customer])
else:
    st.dataframe(segments)

offer = st.sidebar.radio("Sözleşme Teklifi :", ["Yapılmadı", "Yapıldı"])


def user_input_features(df, production):
    state = st.sidebar.selectbox("Eyalet Seç", ["Washington", "Oregon", "Nevada", "California", "Arizona"])
    response = st.sidebar.selectbox("Cevap Seç", ["Yes", "No"])
    coverage = st.sidebar.selectbox("Kapsam Seç", ["Basic", "Extended", "Premium"])
    education = st.sidebar.selectbox("Eğitim Durumu Seç",
                                     ["High School or Below", "College", "Bachelor", "Doctor", "Master"])
    employment = st.sidebar.selectbox("Çalışma Durumu Seç",
                                      ["Employed", "Unemployed", "Medical Leave", "Disabled", "Retired"])
    gender = st.sidebar.selectbox("Cinsiyet Seç", ["M", "F"])
    income = st.sidebar.number_input("Yıllık Geliri Gir", 0, 100000)
    location = st.sidebar.selectbox("Lokasyon Seç", ["Urban", "Suburban", "Rural"])
    marital = st.sidebar.selectbox("Medeni Durumu Seç", ["Married", "Single", "Divorced"])
    premium = st.sidebar.number_input("Aylık Ödenen Parayı Gir", 0, 300)
    claim = st.sidebar.number_input("Son Şikayetten Geçen Süreyi Seç", 0, 50)
    inception = st.sidebar.number_input("Poliçe Başlangıcından Sonra Geçen Süreyi Git", 0, 50)
    complaints = st.sidebar.number_input("Açık Şikayet Sayısını Gir", 0, 5)
    number_policies = st.sidebar.number_input("Poliçe Sayısını Gir", 1, 9)
    policy_type = st.sidebar.selectbox("Poliçe Tipi Seç", ["Corporate Auto", "Personal Auto", "Special Auto"])
    policy = st.sidebar.selectbox("Poliçe Alt Tipi Seç",
                                  ['Corporate L1', 'Corporate L2', 'Corporate L3', 'Personal L1', 'Personal L2',
                                   'Personal L3', 'Special L1', 'Special L2', 'Special L3'])
    offer = st.sidebar.selectbox("Teklifi Seç", ["Offer1", "Offer2", "Offer3", "Offer4"])
    sales = st.sidebar.selectbox("Satış Kanalını Seç", ["Agent", "Call Center", "Web", "Branch"])
    claim_amount = st.sidebar.number_input("Ödenen Sigorta Parasını gir", 0, 3000)
    vehicle_class = st.sidebar.selectbox("Araç Sınıfını Seç",
                                         ["Two-Door Car", "Four-Door Car", "SUV", "Luxury SUV", "Sports Car",
                                          "Luxury Car"])
    vehicle_size = st.sidebar.selectbox("Araç Boyutunu Seç", ["Small", "Medsize", "Large"])

    data = {"State": state,
            "Response": response,
            "Coverage": coverage,
            "Education": education,
            "EmploymentStatus": employment,
            "Gender": gender,
            "Income": income,
            "Location Code": location,
            "Marital Status": marital,
            "Monthly Premium Auto": premium,
            "Months Since Last Claim": claim,
            "Months Since Policy Inception": inception,
            "Number of Open Complaints": complaints,
            "Number of Policies": number_policies,
            "Policy Type": policy_type,
            "Policy": policy,
            "Renew Offer Type": offer,
            "Sales Channel": sales,
            "Total Claim Amount": claim_amount,
            "Vehicle Class": vehicle_class,
            "Vehicle Size": vehicle_size}

    features = pd.DataFrame(data, index=[0])

    pred = pd.concat([df, production], ignore_index=True)

    pred1 = pd.concat([pred, features], ignore_index=True)

    features = cltv_data_preb(pred1)

    return features, premium, response, coverage, number_policies, inception, claim_amount


def user_input_offer():
    state = st.sidebar.selectbox("Eyalet Seç", ["Washington", "Oregon", "Nevada", "California", "Arizona"])
    coverage = st.sidebar.selectbox("Kapsam Seç", ["Basic", "Extended", "Premium"])
    education = st.sidebar.selectbox("Eğitim Durumu Seç",
                                     ["High School or Below", "College", "Bachelor", "Doctor", "Master"])
    employment = st.sidebar.selectbox("Çalışma Durumu Seç",
                                      ["Employed", "Unemployed", "Medical Leave", "Disabled", "Retired"])
    gender = st.sidebar.selectbox("Cinsiyet Seç", ["M", "F"])
    income = st.sidebar.number_input("Yıllık Geliri Gir", 0, 100000)
    location = st.sidebar.selectbox("Lokasyon Seç", ["Urban", "Suburban", "Rural"])
    marital = st.sidebar.selectbox("Medeni Durumu Seç", ["Married", "Single", "Divorced"])
    premium = st.sidebar.number_input("Aylık Ödenen Parayı Gir", 0, 300)
    claim = st.sidebar.number_input("Son Şikayetten Geçen Süreyi Seç", 0, 50)
    inception = st.sidebar.number_input("Poliçe Başlangıcından Sonra Geçen Süreyi Gir", 0, 50)
    complaints = st.sidebar.number_input("Açık Şikayet Sayısını Gir", 0, 5)
    number_policies = st.sidebar.number_input("Poliçe Sayısını Gir", 1, 9)
    policy_type = st.sidebar.selectbox("Poliçe Tipi Seç", ["Corporate Auto", "Personal Auto", "Special Auto"])
    policy = st.sidebar.selectbox("Poliçe Alt Tipi Seç",
                                  ['Corporate L1', 'Corporate L2', 'Corporate L3', 'Personal L1', 'Personal L2',
                                   'Personal L3', 'Special L1', 'Special L2', 'Special L3'])
    claim_amount = st.sidebar.number_input("Ödenen Sigorta Parasını Gir", 0, 3000)
    vehicle_class = st.sidebar.selectbox("Araç Sınıfını Seç",
                                         ["Two-Door Car", "Four-Door Car", "SUV", "Luxury SUV", "Sports Car",
                                          "Luxury Car"])
    vehicle_size = st.sidebar.selectbox("Araç Boyutunu Seç", ["Small", "Medsize", "Large"])

    data = {"State": state,
            "Coverage": coverage,
            "Education": education,
            "EmploymentStatus": employment,
            "Gender": gender,
            "Income": income,
            "Location Code": location,
            "Marital Status": marital,
            "Monthly Premium Auto": premium,
            "Months Since Last Claim": claim,
            "Months Since Policy Inception": inception,
            "Number of Open Complaints": complaints,
            "Number of Policies": number_policies,
            "Policy Type": policy_type,
            "Policy": policy,
            "Total Claim Amount": claim_amount,
            "Vehicle Class": vehicle_class,
            "Vehicle Size": vehicle_size}

    features = pd.DataFrame(data, index=[0])

    return features

c1 = st.container()

with c1:
    if offer == "Yapıldı":
        st.sidebar.header("Müşteri Değerleri")
        features, premium, response, coverage, number_policies, inception, claim_amount = user_input_features(df, production)

        model = joblib.load("rf_final.pkl")
        prediction = model.predict(features)

        st.subheader("Müşteri Bilgileri")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"Müşterinin CLTV'si : {float(prediction.round(2))}")

            if prediction <= 3991.727249:
                st.markdown('Müşterinin Bulunduğu Segment : Bronze')
            elif (prediction > 3991.727249) & (prediction <= 5785.223477):
                st.markdown('Müşterinin Bulunduğu Segment : Silver')
            elif (prediction > 5785.223477) & (prediction <= 9052.951051):
                st.markdown('Müşterinin Bulunduğu Segment : Gold')
            else:
                st.markdown('Müşterinin Bulunduğu Segment : Diamond')

            st.markdown(f"Müşterinin Aylık Ödediği Para : {premium}")

            if response == 'Yes':
                st.markdown("Müşterinin Teklife Verdiği Yanıt : Evet")
            else:
                st.markdown("Müşterinin Teklife Verdiği Yanıt : Hayır")

        with col2:
            st.markdown(f"Sigorta Kapsamı : {coverage}")
            st.markdown(f"Sözleşmedeki Poliçe Sayısı : {number_policies}")
            st.markdown(f"Poliçe Başlangıcından Sonra Geçen Süre : {inception}")
            st.markdown(f"Müşteriye Ödenen Sigorta Parası : {claim_amount}")

        with st.container():
            st.subheader("Müşteriler Hakkında Genel Bilgiler")
            st.markdown(f"Ortalama CLTV Değeri : {round(segments['Customer Lifetime Value'].mean(), 2)}")
            st.markdown(f"Ortalama Ödenen Aylık Ücret : {round(segments['Monthly Premium Auto'].mean(), 2)}")
            st.markdown(
                f"Müşteriye Ödenen Ortalama Sigorta Parası : {round(segments['Total Claim Amount'].mean(), 2)}")
            st.markdown(f"Ortalama Yıllık Gelir : {round(segments['Income'].mean(), 2)}")
            st.markdown(
                f"Ortalama Sigorta Başvurusundan Sonra Geçen Ay Sayısı : {int(segments['Months Since Last Claim'].mean())}")
            st.markdown(
                f"Ortalama Sigorta Başlangıcından Sonra Geçen Ay Sayısı : {int(segments['Months Since Policy Inception'].mean())}")

        st.subheader("Segmentlere Ait Grafikler")

        segments_click = st.selectbox("Segment Seçiniz", ["Diamond", "Gold", "Silver", "Bronze"])

        if segments_click == "Diamond":

            data = pd.DataFrame({"state": ["4", "6", "32", "41", "53"],
                                 "StateName": ["Arizona", "California", "Nevada", "Oregon", "Washington"],
                                 "NumberofCustomer": ["403", "808", "210", "666", "197"]})

            data['text'] = data['StateName'] + '<br> ' + \
                           'Müşteri Sayısı: ' + data['NumberofCustomer']

            fig = go.Figure(data=go.Choropleth(
                locations=["AZ", "CA", "NV", "OR", "WA"],
                z=data['NumberofCustomer'].astype(float),
                locationmode='USA-states',
                colorscale='Sunset',
                text=data['text'],
                marker_line_color='white',
                colorbar_title="Müşteri Sayısı"
            ))

            fig.update_layout(
                title_text='Eyaletlere Göre Diamond Müşteri Sayısı',
                geo=dict(
                    scope='usa',
                    projection=go.layout.geo.Projection(type='albers usa'),
                    showlakes=False,
                    bgcolor='rgba(0,0,0,0)'),
            )

            st.write(fig)

            col3, col4 = st.columns(2)
            col5, col6 = st.columns(2)
            col7, col8 = st.columns(2)

            diamond = segments[segments["Segments"] == "Diamond"]

            with col3:
                st.markdown("Müşteri Yaşam Boyu Değeri :")
                fig = px.histogram(diamond, x="Response", y="Customer Lifetime Value", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col4:
                st.markdown("Araç Sınıfı :")
                fig = px.histogram(diamond, x="Vehicle Class", color="Vehicle Class", pattern_shape="Segments")
                fig.update_layout(width=500)
                st.write(fig)

            with col5:
                st.markdown("Poliçe Sayısı :")
                fig = px.histogram(diamond, x="Number of Policies", color="Number of Policies",
                                   pattern_shape="Segments")
                fig.update_layout(width=500)
                st.write(fig)

            with col6:
                st.markdown("Aylık Ödenen Ücret :")
                fig = px.histogram(diamond, x="Response", y="Monthly Premium Auto", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col7:
                st.markdown("Toplam Ödenen Sigorta Parası :")
                fig = px.histogram(diamond, x="Response", y="Total Claim Amount", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col8:
                st.markdown("Sigorta Kapsamı :")
                fig = px.histogram(diamond, x="Coverage", pattern_shape="Response")
                fig.update_layout(width=500)
                st.write(fig)


        elif segments_click == "Gold":

            data = pd.DataFrame({"state": ["4", "6", "32", "41", "53"],
                                 "StateName": ["Arizona", "California", "Nevada", "Oregon", "Washington"],
                                 "NumberofCustomer": ["452", "793", "212", "661", "165"]})

            data['text'] = data['StateName'] + '<br> ' + \
                           'Müşteri Sayısı: ' + data['NumberofCustomer']

            fig = go.Figure(data=go.Choropleth(
                locations=["AZ", "CA", "NV", "OR", "WA"],
                z=data['NumberofCustomer'].astype(float),
                locationmode='USA-states',
                colorscale='Sunset',
                text=data['text'],
                marker_line_color='white',
                colorbar_title="Müşteri Sayısı"
            ))

            fig.update_layout(
                title_text='Eyaletlere Göre Gold Müşteri Sayısı',
                geo=dict(
                    scope='usa',
                    projection=go.layout.geo.Projection(type='albers usa'),
                    showlakes=False,
                    bgcolor='rgba(0,0,0,0)'),
            )

            st.write(fig)

            col3, col4 = st.columns(2)
            col5, col6 = st.columns(2)
            col7, col8 = st.columns(2)

            diamond = segments[segments["Segments"] == "Gold"]

            with col3:
                st.markdown("Müşteri Yaşam Boyu Değeri :")
                fig = px.histogram(diamond, x="Response", y="Customer Lifetime Value", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col4:
                st.markdown("Araç Sınıfı :")
                fig = px.histogram(diamond, x="Vehicle Class", color="Vehicle Class", pattern_shape="Segments")
                fig.update_layout(width=500)
                st.write(fig)

            with col5:
                st.markdown("Poliçe Sayısı :")
                fig = px.histogram(diamond, x="Number of Policies", color="Number of Policies",
                                   pattern_shape="Segments")
                fig.update_layout(width=500)
                st.write(fig)

            with col6:
                st.markdown("Aylık Ödenen Ücret :")
                fig = px.histogram(diamond, x="Response", y="Monthly Premium Auto", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col7:
                st.markdown("Toplam Ödenen Sigorta Parası :")
                fig = px.histogram(diamond, x="Response", y="Total Claim Amount", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

                with col8:
                    st.markdown("Sigorta Kapsamı :")
                    fig = px.histogram(diamond, x="Coverage", pattern_shape="Response")
                    fig.update_layout(width=500)
                    st.write(fig)


        elif segments_click == "Silver":

            data = pd.DataFrame({"state": ["4", "6", "32", "41", "53"],
                                 "StateName": ["Arizona", "California", "Nevada", "Oregon", "Washington"],
                                 "NumberofCustomer": ["425", "769", "229", "648", "212"]})

            data['text'] = data['StateName'] + '<br> ' + \
                           'Müşteri Sayısı: ' + data['NumberofCustomer']

            fig = go.Figure(data=go.Choropleth(
                locations=["AZ", "CA", "NV", "OR", "WA"],
                z=data['NumberofCustomer'].astype(float),
                locationmode='USA-states',
                colorscale='Sunset',
                text=data['text'],
                marker_line_color='white',
                colorbar_title="Müşteri Sayısı"
            ))

            fig.update_layout(
                title_text='Eyaletlere Göre Silver Müşteri Sayısı',
                geo=dict(
                    scope='usa',
                    projection=go.layout.geo.Projection(type='albers usa'),
                    showlakes=False,
                    bgcolor='rgba(0,0,0,0)'),
            )

            st.write(fig)

            col3, col4 = st.columns(2)
            col5, col6 = st.columns(2)
            col7, col8 = st.columns(2)

            diamond = segments[segments["Segments"] == "Silver"]

            with col3:
                st.markdown("Müşteri Yaşam Boyu Değeri :")
                fig = px.histogram(diamond, x="Response", y="Customer Lifetime Value", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col4:
                st.markdown("Araç Sınıfı :")
                fig = px.histogram(diamond, x="Vehicle Class", color="Vehicle Class", pattern_shape="Segments")
                fig.update_layout(width=500)
                st.write(fig)

            with col5:
                st.markdown("Poliçe Sayısı :")
                fig = px.histogram(diamond, x="Number of Policies", color="Number of Policies",
                                   pattern_shape="Segments")
                fig.update_layout(width=500)
                st.write(fig)

            with col6:
                st.markdown("Aylık Ödenen Ücret :")
                fig = px.histogram(diamond, x="Response", y="Monthly Premium Auto", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col7:
                st.markdown("Toplam Ödenen Sigorta Parası :")
                fig = px.histogram(diamond, x="Response", y="Total Claim Amount", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col8:
                st.markdown("Sigorta Kapsamı :")
                fig = px.histogram(diamond, x="Coverage", pattern_shape="Response")
                fig.update_layout(width=500)
                st.write(fig)


        else:

            data = pd.DataFrame({"state": ["4", "6", "32", "41", "53"],
                                 "StateName": ["Arizona", "California", "Nevada", "Oregon", "Washington"],
                                 "NumberofCustomer": ["423", "780", "231", "626", "224"]})

            data['text'] = data['StateName'] + '<br> ' + \
                           'Müşteri Sayısı: ' + data['NumberofCustomer']

            fig = go.Figure(data=go.Choropleth(
                locations=["AZ", "CA", "NV", "OR", "WA"],
                z=data['NumberofCustomer'].astype(float),
                locationmode='USA-states',
                colorscale='Sunset',
                text=data['text'],
                marker_line_color='white',
                colorbar_title="Müşteri Sayısı"
            ))

            fig.update_layout(
                title_text='Eyaletlere Göre Bronze Müşteri Sayısı',
                geo=dict(
                    scope='usa',
                    projection=go.layout.geo.Projection(type='albers usa'),
                    showlakes=False,
                    bgcolor='rgba(0,0,0,0)'),
            )

            st.write(fig)

            col3, col4 = st.columns(2)
            col5, col6 = st.columns(2)
            col7, col8 = st.columns(2)

            diamond = segments[segments["Segments"] == "Bronze"]

            with col3:
                st.markdown("Müşteri Yaşam Boyu Değeri :")
                fig = px.histogram(diamond, x="Response", y="Customer Lifetime Value", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col4:
                st.markdown("Araç Sınıfı :")
                fig = px.histogram(diamond, x="Vehicle Class", color="Vehicle Class", pattern_shape="Segments")
                fig.update_layout(width=500)
                st.write(fig)

            with col5:
                st.markdown("Poliçe Sayısı :")
                fig = px.histogram(diamond, x="Number of Policies", color="Number of Policies",
                                   pattern_shape="Segments")
                fig.update_layout(width=500)
                st.write(fig)

            with col6:
                st.markdown("Aylık Ödenen Ücret :")
                fig = px.histogram(diamond, x="Response", y="Monthly Premium Auto", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col7:
                st.markdown("Toplam Ödenen Sigorta Parası :")
                fig = px.histogram(diamond, x="Response", y="Total Claim Amount", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            with col8:
                st.markdown("Sigorta Kapsamı :")
                fig = px.histogram(diamond, x="Coverage", pattern_shape="Response")
                fig.update_layout(width=500)
                st.write(fig)

        st.subheader("Segment Karşılaştırma")
        options = st.multiselect("Karşılaştırılmak İstenen Segmentleri Seçiniz :",
                                 ["Diamond", "Gold", "Silver", "Bronze"])

        if len(options) > 1:
            if len(options) == 2:
                col9, col10 = st.columns(2)
                compare = segments[(segments["Segments"] == options[0]) | (segments["Segments"] == options[1])]
                with col9:
                    st.markdown("Müşteri Yaşam Boyu Değeri :")
                    fig = px.histogram(compare, x="Segments", y="Customer Lifetime Value", color="Segments",
                                       histfunc="avg")
                    fig.update_layout(width=500)
                    st.write(fig)
                with col10:
                    st.markdown("Aylık Ödenen Ücret :")
                    fig = px.histogram(compare, x="Segments", y="Monthly Premium Auto", color="Segments",
                                       histfunc="avg")
                    fig.update_layout(width=500)
                    st.write(fig)

                st.markdown("Toplam Ödenen Sigorta Parası :")
                fig = px.histogram(compare, x="Segments", y="Total Claim Amount", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            elif len(options) == 3:
                col9, col10 = st.columns(2)
                compare = segments[
                    (segments["Segments"] == options[0]) | (segments["Segments"] == options[1]) | (
                            segments["Segments"] == options[2])]
                with col9:
                    st.markdown("Müşteri Yaşam Boyu Değeri :")
                    fig = px.histogram(compare, x="Segments", y="Customer Lifetime Value", color="Segments",
                                       histfunc="avg")
                    fig.update_layout(width=500)
                    st.write(fig)
                with col10:
                    st.markdown("Aylık Ödenen Ücret :")
                    fig = px.histogram(compare, x="Segments", y="Monthly Premium Auto", color="Segments",
                                       histfunc="avg")
                    fig.update_layout(width=500)
                    st.write(fig)

                st.markdown("Toplam Ödenen Sigorta Parası :")
                fig = px.histogram(compare, x="Segments", y="Total Claim Amount", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

            else:
                col9, col10 = st.columns(2)
                with col9:
                    st.markdown("Müşteri Yaşam Boyu Değeri :")
                    fig = px.histogram(segments, x="Segments", y="Customer Lifetime Value", color="Segments",
                                       histfunc="avg")
                    fig.update_layout(width=500)
                    st.write(fig)
                with col10:
                    st.markdown("Aylık Ödenen Ücret :")
                    fig = px.histogram(segments, x="Segments", y="Monthly Premium Auto", color="Segments",
                                       histfunc="avg")
                    fig.update_layout(width=500)
                    st.write(fig)

                st.markdown("Toplam Ödenen Sigorta Parası :")
                fig = px.histogram(segments, x="Segments", y="Total Claim Amount", color="Segments",
                                   histfunc="avg")
                fig.update_layout(width=500)
                st.write(fig)

    elif offer == "Yapılmadı":
        st.sidebar.header("Müşteri Değerleri")
        features = user_input_offer()

        button1 = st.sidebar.button("Teklif Öner")

        if button1 == True:
            st.subheader("Önerilen Teklif :")
            st.markdown("Müşteri Bilgileri :")
            st.dataframe(features)

            if (features.loc[0, "Marital Status"] == "Married") and (features.loc[0, "EmploymentStatus"] == "Employed" or features.loc[0, "EmploymentStatus"] == "Retired"):
                st.markdown("Bu müşteriye **Teklif 3'ün** önerilmesi tavsiye edilir.")
            elif (features.loc[0, "Number of Open Complaints"] > 2) or (features.loc[0, "Vehicle Class"] in ["Two-Door Car", "Four-Door Car", "Sports Car"]):
                st.markdown("Bu müşteriye **Teklif 1'in** önerilmesi tavsiye edilir.")
            elif (features.loc[0, "Number of Open Complaints"] <= 2) or (features.loc[0, "Vehicle Class"] in ["Luxury Car", "Luxury SUV"]):
                st.markdown("Bu müşteriye **Teklif 2'nin** önerilmesi tavsiye edilir.")
            else:
                st.markdown("**Yeni bir teklif düşünülmesi tavsiye edilir.**")
