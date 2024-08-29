import streamlit as st

# Initialize session state to manage authentication
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

def main():
    # Check if user is authenticated
    if st.session_state['authentication_status'] is None:
        st.title("Login Page")

        # Login form
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "admin":
                st.session_state['authentication_status'] = 'admin'
                st.success("Welcome Admin!")
            elif username == "user" and password == "user":
                st.session_state['authentication_status'] = 'user'
                st.success("Welcome User!")
            else:
                st.error("Invalid username or password")

        # Spacer to push the developer section to the bottom
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
        # Developer Information
        st.markdown("---")
        st.subheader("Developed by:")
        col1, col2, col3 = st.columns([1, 1, 1], gap="small")

        with col1:
            st.image("pho.jpg", caption="Harshit", width=80)
        
        with col2:
            st.image("pho.jpg", caption="Sharath", width=80)
        
        with col3:
            st.image("pho.jpg", caption="Chakradhar", width=80)

    elif st.session_state['authentication_status'] == 'admin':
        st.title("Admin Page")
        import app  # Import and run app.py content directly here
        app.run()  # This assumes you have a function `run()` in app.py
    elif st.session_state['authentication_status'] == 'user':
        st.title("User Page")
        import appp  # Import and run appp.py content directly here
        appp.run()  # This assumes you have a function `run()` in appp.py

if __name__ == "__main__":
    main()
