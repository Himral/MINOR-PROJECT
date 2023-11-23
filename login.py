import streamlit as st
import mysql.connector

def authenticate_user(username, password):
    try:
        
        connection = mysql.connector.connect(
            host="localhost",
            user="root",  
            password="",  
            database="user"  
        )

        cursor = connection.cursor()

        # Query to check if the user exists in the database
        query = f"SELECT * FROM user WHERE userName = '{username}' AND password = '{password}'"
        cursor.execute(query)

        user = cursor.fetchone()

        if user:
            return True
        else:
            return False

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def add_user(username, password):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",  
            password="",  
            database="user"  
        )

        cursor = connection.cursor()
        query = f"INSERT INTO user (userName, password) VALUES ('{username}', '{password}')"
        cursor.execute(query)

        connection.commit()
        st.success("User successfully added to the database. You can now log in.")

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def main():
    st.title("Login Page with XAMPP Database")
    st.sidebar.title("Sign Up")
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")
    sign_up_button = st.sidebar.button("Sign Up")
    st.write("## Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")
    # Handle sign-up
    if sign_up_button:
        if new_username and new_password:
            add_user(new_username, new_password)
    # Handle login
    if login_button:
        if username and password:
            if authenticate_user(username, password):
                st.success("Login successful!")
                new_app_url = "http://localhost:8504"  # Replace with the URL of your other Streamlit app
                st.experimental_set_query_params(next_page=new_app_url)
            else:
                st.error("Invalid username or password. Please try again.")

if __name__ == "__main__":
    main()
