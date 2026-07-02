import streamlit as st

pg = st.navigation([
    st.Page("pages/0_Home.py", title="🏠 Home"),
    st.Page("pages/1_Data.py", title="🔬 Data"),
    st.Page("pages/2_Preprocessing.py", title="⚙️ Preprocessing"),
    st.Page("pages/3_Models.py", title="🧠 Models"),
    st.Page("pages/4_Interactive_classification.py", title="🔍 Interactive classification"),
    #st.Page("pages/5_Conclusion.py", title="💡 Conclusion"),
])

pg.run()
