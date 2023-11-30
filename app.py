import nltk
import streamlit as st
import re
import preprocessor,helper
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    layout="wide",
    page_title="Whatsapp Chat Analyzer",
    )
st.sidebar.title("Whatsapp Chat Analyzer")

nltk.download('vader_lexicon')
nltk.download('wordnet')

uploaded_file = st.sidebar.file_uploader("Choose a file")

st. markdown("<h1 style='text-align: center; color: green;'>Whatsapp Chat Analyzer</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    bytes_df = uploaded_file.getvalue()
    data = bytes_df.decode("utf-8")
    df = preprocessor.preprocess(data)
    
    st.dataframe(df)

    #fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')

    user_list.sort()
    user_list.insert(0,"Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if(st.sidebar.button("Show analysis")):

        num_messages, words,num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        col1 , col2 , col3, col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        #monthly timeline quantitative
        st.markdown("<h2 style='text-align: center; color: green;'>Quantitative Analysis: Monthly Timeline</h2>", unsafe_allow_html=True)
        st.subheader("")
        st.write("  ")

        timeline = helper.monthly_timeline(selected_user, df)
        #timeline['time'] = pd.to_datetime(timeline['time'])
        #timeline['month_year'] = timeline['time'].dt.to_period('M')
  # Create a new column combining month and year
        #timeline = timeline.sort_values('month_year')

        #st.dataframe(timeline)
        st.line_chart(timeline, x='time', y='message')

        
        

        from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
        # Object
        sentiments = SentimentIntensityAnalyzer()
    
        # Creating different columns for (Positive/Negative/Neutral)
        df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]] # Positive
        df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]] # Negative
        df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]] # Neutral
    
        # To indentify true sentiment per row in message column
        def sentiment(d):
            if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
                return 1
            if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
                return -1
            if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
                return 0

        # Creating new column & Applying function
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)
        # Monthly timeline Sentiment
        st. markdown("<h2 style='text-align: center; color: green;'>Sentiment Analysis : Monthly Timeline</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Positive</h3>",unsafe_allow_html=True)
                
            timeline = helper.sentiment_monthly_timeline(selected_user, df,1)
                
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Neutral</h3>",unsafe_allow_html=True)
                
            timeline = helper.sentiment_monthly_timeline(selected_user, df,0)
                
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Negative</h3>",unsafe_allow_html=True)
                
            timeline = helper.sentiment_monthly_timeline(selected_user, df,-1)
                
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Percentage contributed
        if selected_user == 'Overall':
            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",unsafe_allow_html=True)
                x = helper.sentiment_percentage(df, 1)
                    
                    # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",unsafe_allow_html=True)
                y = helper.sentiment_percentage(df, 0)
                    
                    # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",unsafe_allow_html=True)
                z = helper.sentiment_percentage(df, -1)
                    
                    # Displaying
                st.dataframe(z)
        #daily timeline
        sns.set_style("darkgrid")
        st. markdown("<h2 style='text-align: center; color: green;'>Quantitative Analysis : Daily Timeline</h2>", unsafe_allow_html=True)
        daily_timeline=helper.daily_timeline(selected_user,df)
        fig,ax=plt.subplots()
        st.line_chart(daily_timeline,x = "only_date",y = "message")
        ax.plot(daily_timeline['only_date'],daily_timeline['message'],color='brown')
        plt.xticks(rotation='vertical')
        #st.pyplot(fig)

    # Daily timeline Sentiment
        st. markdown("<h2 style='text-align: center; color: green;'>Sentiment Analysis : Daily Timeline</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Positive</h3>",unsafe_allow_html=True)
                
            daily_timeline = helper.sentiment_daily_timeline(selected_user, df, 1)
                
            fig, ax = plt.subplots()
            st.line_chart(daily_timeline,x = "only_date",y = "message")
            ax.plot(daily_timeline['only_date'],daily_timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            #st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Neutral</h3>",unsafe_allow_html=True)
                
            daily_timeline = helper.sentiment_daily_timeline(selected_user, df, 0)
                
            fig, ax = plt.subplots()
            st.line_chart(daily_timeline,x = "only_date",y = "message")
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
            plt.xticks(rotation='vertical')
            #st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Negative</h3>",unsafe_allow_html=True)
                
            daily_timeline = helper.sentiment_daily_timeline(selected_user, df, -1)
                
            fig, ax = plt.subplots()
            st.line_chart(daily_timeline,x = "only_date",y = "message")
            ax.plot(daily_timeline['only_date'],daily_timeline['message'], color='red')
            plt.xticks(rotation='vertical')
            #st.pyplot(fig)
            
        #most active users (in group)
        if selected_user == 'Overall':
            st.markdown("<h2 style='text-align: center; color: green;'>Quantitative Analysis: Most Active Users</h2>",unsafe_allow_html=True)
            st.subheader("")
            st.write("  ")
            x, new_df =helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)
            
            with col1:
                name = x.index
                count = x.values
                ax.bar(name,count,color = 'purple')
                plt.xticks(rotation = 'vertical')
                st.pyplot(fig)
            
            with col2:
                st.dataframe(new_df)
                
        if selected_user == 'Overall':
                
                # Getting names per sentiment
            x = df['user'][df['value'] == 1].value_counts().head(10)
            y = df['user'][df['value'] == -1].value_counts().head(10)
            z = df['user'][df['value'] == 0].value_counts().head(10)
            st.markdown("<h2 style='text-align: center; color: green;'>Sentiment Analysis: Most Active users</h2>",unsafe_allow_html=True)
            st.subheader("")
            st.write("  ")
            col1,col2,col3 = st.columns(3)
            with col1:
                    # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Users</h3>",unsafe_allow_html=True)
                    
                    # Displaying
                fig, ax = plt.subplots()
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                    # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Users</h3>",unsafe_allow_html=True)
                    
                    # Displaying
                fig, ax = plt.subplots()
                ax.bar(z.index, z.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                    # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",unsafe_allow_html=True)
                    
                    # Displaying
                fig, ax = plt.subplots()
                ax.bar(y.index, y.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)


        #activity map
        st.markdown("<h2 style='text-align: center; color: green;'>Quantitative Analysis: Most Busy Month</h2>",unsafe_allow_html=True)
        st.subheader("")
        st.write("  ")
        col1,col2=st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day=helper.week_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month=helper.month_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_month.index,busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Positive)</h3>",unsafe_allow_html=True)
                
            busy_day = helper.sentiment_week_activity_map(selected_user, df,1)
                
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values,color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Neutral)</h3>",unsafe_allow_html=True)
                
            busy_day = helper.sentiment_week_activity_map(selected_user, df, 0)
                
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Negative)</h3>",unsafe_allow_html=True)
                
            busy_day = helper.sentiment_week_activity_map(selected_user, df, -1)
                
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Positive)</h3>",unsafe_allow_html=True)
                
            busy_month = helper.sentiment_month_activity_map(selected_user, df,1)
                
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",unsafe_allow_html=True)
                
            busy_month = helper.sentiment_month_activity_map(selected_user, df, 0)
                
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='grey')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",unsafe_allow_html=True)
                
            busy_month = helper.sentiment_month_activity_map(selected_user, df, -1)
                
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        st.markdown("<h2 style='text-align: center; color: green;'>Quantitative Analysis: Most Active Hour of the Week</h2>",unsafe_allow_html=True)
        st.subheader("")
        st.write("  ")
        user_heatmap=helper.activity_heatmap(selected_user,df)
        fig,ax=plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)
        st.markdown("<h2 style='text-align: center; color: green;'>Sentiment Analysis: Most Active Hour of the Week</h2>",unsafe_allow_html=True)
        st.subheader("")
        st.write("  ")
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Positive)</h3>",unsafe_allow_html=True)
                    
                user_heatmap = helper.sentiment_activity_heatmap(selected_user, df, 1)
                    
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col2:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Neutral)</h3>",unsafe_allow_html=True)
                    
                user_heatmap = helper.sentiment_activity_heatmap(selected_user, df, 0)
                    
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col3:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Negative)</h3>",unsafe_allow_html=True)
                    
                user_heatmap = helper.sentiment_activity_heatmap(selected_user, df, -1)
                    
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                st.pyplot(fig)
            except:
                st.image('error.webp')
        #wordcloud
        st.markdown("<h2 style='text-align: center; color: green;'>Quantitative Analysis: Wordcloud</h2>",unsafe_allow_html=True)
        st.subheader("")
        st.write("  ")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)
        
        st.markdown("<h2 style='text-align: center; color: green;'>Sentiment Analysis: Wordcloud</h2>",unsafe_allow_html=True)
        st.subheader("")
        st.write("  ")
        col1,col2,col3 = st.columns(3)
        with col1:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Positive WordCloud</h3>",unsafe_allow_html=True)
                    
                # Creating wordcloud of positive words
                df_wc = helper.sentiment_create_wordcloud(selected_user, df,1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                # Display error message
                st.image('error.webp')
        with col2:
            try:
                    # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral WordCloud</h3>",unsafe_allow_html=True)
                    
                    # Creating wordcloud of neutral words
                df_wc = helper.sentiment_create_wordcloud(selected_user, df,0)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                    # Display error message
                st.image('error.webp')
        with col3:
            try:
                    # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Negative WordCloud</h3>",unsafe_allow_html=True)
                    
                    # Creating wordcloud of negative words
                df_wc = helper.sentiment_create_wordcloud(selected_user, df,-1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                st.pyplot(fig)
            except:
                    # Display error message
                st.image('error.webp')
        #Frequent Words
        most_common_df= helper.most_common_words(selected_user,df)
        
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation = 'vertical')
        st.markdown("<h2 style='text-align: center; color: green;'>Quantitative Analysis: Most Frequent Words</h2>",unsafe_allow_html=True)
        st.subheader("")
        st.write("  ")
        st.pyplot(fig)
        st.markdown("<h2 style='text-align: center; color: green;'>Sentiment Analysis: Most Frequent Words</h2>",unsafe_allow_html=True)
        st.subheader("")
        st.write("  ")
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                    # df frame of most common positive words.
                most_common_df = helper.sentiment_most_common_words(selected_user, df,1)
                    
                    # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Positive Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1],color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                    # Disply error image
                st.image('error.webp')
        with col2:
            try:
                    # df frame of most common neutral words.
                most_common_df = helper.sentiment_most_common_words(selected_user, df,0)
                    
                    # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1],color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                    # Disply error image
                st.image('error.webp')
        with col3:
            try:
                    # df frame of most common negative words.
                most_common_df = helper.sentiment_most_common_words(selected_user, df,-1)
                    
                    # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Negative Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                    # Disply error image
                st.image('error.webp')     
        
            # Monthly activity map
        

            # Daily activity map
        

        

        
        
            

