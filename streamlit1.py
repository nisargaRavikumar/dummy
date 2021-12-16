import streamlit as st
import joblib
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
model = joblib.load('finalized_model1.sav')
X_test= pd.read_csv(r'X_test.csv')
test= pd.read_csv(r'IFDtest.csv',parse_dates=['Transaction_date'])
book_ids=test['book_id'].unique()
result = model.predict(X_test, num_iteration=model.best_iteration)
forecast = pd.DataFrame({"date":test["Transaction_date"],
                        "store":test["store_id"],
                        "item":test["book_id"],
                        "sales":result
                        })      
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Books Sales Forecasting</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Store = st.selectbox('Store',(1,2,3,4,5,6,7,8,9,10))
    Item = st.selectbox('Item',tuple(book_ids)) 
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        df = forecast[(forecast.store == Store) & (forecast.item == Item)]                        
        fig = px.line(        
            df, #Data Frame
            x = "date", #Columns from the data frame
            y = "sales",
            title = "Line frame"
        )
        st.write(fig)
     
if __name__=='__main__': 
    main()
