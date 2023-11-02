from urlextract import URLExtract
extract = URLExtract()

def fetch_stats(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user']== selected_user]
    #fetch the number of messages

    num_messages = df.shape[0]

    #fetch no. of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    #fetch total media messages
    num_media_messages=df[df['message'] == '<Media omitted>\n'].shape[0]
    
    #fetching link
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    return x

    