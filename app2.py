import streamlit as st
import pickle
import numpy as np
from sklearn import preprocessing 
import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# import the model
pipe = pickle.load(open('pipe3.pkl','rb'))
# df = pickle.load(open('df3.pkl','rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Laptop Predictor")

balance=st.number_input(label='Balance',step=0.001,format="%.6f")
    
balance_frequency=st.number_input(label='Balance Frequency',step=0.001,format="%.6f")
purchases=st.number_input(label='Purchases',step=0.01,format="%.2f")
oneoff_purchases=st.number_input(label='OneOff_Purchases',step=0.01,format="%.2f")
installments_purchases=st.number_input(label='Installments Purchases',step=0.01,format="%.2f")
cash_advance=st.number_input(label='Cash Advance',step=0.01,format="%.6f")
purchases_frequency=st.number_input(label='Purchases Frequency',step=0.01,format="%.6f")
oneoff_purchases_frequency=st.number_input(label='OneOff Purchases Frequency',step=0.1,format="%.6f")
purchases_installment_frequency=st.number_input(label='Purchases Installments Freqency',step=0.1,format="%.6f")
cash_advance_frequency=st.number_input(label='Cash Advance Frequency',step=0.1,format="%.6f")
cash_advance_trx=st.number_input(label='Cash Advance Trx',step=0.1)
purchases_trx=st.number_input(label='Purchases TRX',step=0.1)
credit_limit=st.number_input(label='Credit Limit',step=0.1,format="%.1f")
payments=st.number_input(label='Payments',step=0.01,format="%.6f")
minimum_payments=st.number_input(label='Minimum Payments',step=0.01,format="%.6f")
prc_full_payment=st.number_input(label='PRC Full Payment',step=0.01,format="%.6f")
tenure=st.number_input(label='Tenure',step=0.1)

if st.button('Predict Price'):
 query = np.array([balance,balance_frequency,purchases,oneoff_purchases,installments_purchases,cash_advance,purchases_frequency,oneoff_purchases_frequency,purchases_installment_frequency,cash_advance_frequency,cash_advance_trx,purchases_trx,credit_limit,payments,minimum_payments,prc_full_payment,tenure])

 query = query.reshape(1,17)
    
 cluster=int(pipe.predict(query))
 st.title(cluster)
 


 
