import gzip
import os
import pickle

import pandas as pd  # noqa: F401  -- required at runtime to unpickle the DataFrame
import requests
import streamlit as st
from PIL import Image

# Page configuration

st.set_page_config(layout="wide",initial_sidebar_state="expanded")

# CSS Styling

with open('style.css') as f:

    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 


# navbar


st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">'
'<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous"></script>', unsafe_allow_html=True)



st.markdown(""" 

<nav class="navbar  navbar-expand-lg navbar-dark bg-dark" style=" ">
  <div class="container-fluid">
    <div class="d-flex justify-content-between w-100">
      <a class="navbar-brand d-flex align-items-center" href="https://whatowatchnext.streamlit.app/">
        <img src="https://th.bing.com/th/id/OIP.ZKF5kSelg-1ogpHrJUOAXgHaEK?rs=1&pid=ImgDetMain" alt="Logo" width="60" height="40" class="d-inline-block align-self-center brand-logo">
        <b class="text123 ml-2 brand-text">Movies Recommender</b>
      </a>
      <div class="dropdown"><button class="navbar-toggler" type="button" data-bs-toggle="dropdown" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <ul class="dropdown-menu dropdown-menu-end">
        <li><a class="dropdown-item" href="https://www.youtube.com/@Xx-Black_Reaper001-xX" target="_blank"><img src="https://cdn.cultofmac.com/wp-content/uploads/2018/01/YouTube-dark-780x390.jpg" alt="YouTube Logo" width="100" height="70"></a></li>
        <li><a class="dropdown-item" href="https://twitter.com/Xx_Sujal_xX" target="_blank"><img src="https://th.bing.com/th?id=OSAAS.5E8A0E30002329042FCEC75E8A2561C7&w=80&h=80&c=1&rs=1&o=6&dpr=1.7&pid=5.1" alt="Twitter Logo" width="60" height="60" class="rounded-circle twitter-logo"></a></li>
      </ul>
    </div></div>
    <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
      <ul class="navbar-nav">
        <li class="nav-item">
          <a class="nav-link" href="https://github.com/xxwizardxx117" target="_blank"><img src="https://seeklogo.com/images/G/github-logo-2E3852456C-seeklogo.com.png" alt="Github Logo" width="60" height="60" class="rounded-circle Git-logo"> </a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="https://www.youtube.com/@Xx-Black_Reaper001-xX" target="_blank"><img src="https://i.pinimg.com/originals/cb/99/48/cb99488d42a3d1b7c1061e1ed6cedd58.jpg" alt="YouTube Logo" width="60" height="60"class="rounded-circle yt-logo "></a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="https://twitter.com/Xx_Sujal_xX" target="_blank"><img src="https://th.bing.com/th?id=OSAAS.5E8A0E30002329042FCEC75E8A2561C7&w=80&h=80&c=1&rs=1&o=6&dpr=1.7&pid=5.1" alt="Twitter Logo" width="60" height="60" class="rounded-circle twitter-logo"> </a>
        </li>
      </ul>
    </div>
  </div>
</nav>

<style>

@media (max-width: 690px) {
  .navbar {
    flex-direction: column;
  }
  .navbar .container-fluid {
    flex-direction: column;
  }
  .navbar .dropdown-menu {
    position: static;
    transform: none;
  }
  .brand-logo {
    width: 50px;
    height: 30px;
  }
  .brand-text {
    font-size: 16px;
  }
}

@media (max-width: 428px) {
  .navbar {
    margin-left: 10px;
    margin-right: 10px;
    width: calc(100% - 20px);
  }
  .brand-logo {
    width: 40px;
    height: 20px;
  }
  .brand-text {
    font-size: 14px;
  }
}

@media (min-width: 429px) and (max-width: 690px) {
  .navbar {
    margin-left: 20px;
    margin-right: 20px;
    width: calc(100% - 40px);
  }




}
</style>
""", unsafe_allow_html=True)






# File Imports and Renaming



# # both are old ways
# with gzip.open('movie_list.pkl.gz', 'rb') as f:
#     df_import_list = pickle.load(f)


# # df_import_list = pickle.load(open('movie_list.pkl','rb'))
# #df_import_list = df_import_list['title']    this is a list and the funtion related to it is not working 

# df_import_list = pd.DataFrame(df_import_list)

# # print(df_import_list)

# # Load 'Similarity' from a gzip compressed pickle file
# with gzip.open('similarity.pkl.gz', 'rb') as f:
#     Similarity = pickle.load(f)
# # Similarity = pickle.load(open('similarity.pkl','rb'))



# way 3 

# Streamlit re-executes this entire script on every widget interaction. Without
# a cache, the 47 MB similarity matrix is un-gzipped and unpickled every single
# time. @st.cache_resource loads it once per server process and hands back the
# same object thereafter.
#
# Note: pandas stays imported above even though `pd` is no longer referenced
# directly -- unpickling a DataFrame requires pandas to be importable, so the
# dependency is real. Do not remove it from pyproject.toml/requirements.txt.

@st.cache_resource
def load_artifacts():
    with gzip.open('movie_list.pkl.gz', 'rb') as f:
        movies = pickle.load(f)
    with gzip.open('similarity.pkl.gz', 'rb') as f:
        similarity = pickle.load(f)
    return movies, similarity


df_import_list, Similarity = load_artifacts()


# retry api request
def make_request_with_retry(url, params=None, max_retries=3):
    for _ in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=(3.05, 10))
            response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
            return response
        except requests.exceptions.ConnectionError:
            continue
    raise requests.exceptions.ConnectionError(
        f"Failed to connect to {url} after {max_retries} attempts"
    )



# TMDB credentials

# The key is never committed. Locally it comes from .streamlit/secrets.toml
# (gitignored) or a TMDB_API_KEY environment variable; on Streamlit Cloud it
# comes from the app's Settings -> Secrets panel.

def load_tmdb_api_key():

    try:
        key = st.secrets['TMDB_API_KEY']
    except (KeyError, FileNotFoundError):
        key = os.environ.get('TMDB_API_KEY', '')

    if not key:
        st.error(
            'TMDB_API_KEY is not configured. Add it to .streamlit/secrets.toml '
            'or set it as an environment variable, then reload the app.'
        )
        st.stop()

    return key


TMDB_API_KEY = load_tmdb_api_key()


# Poster fetching function
@st.cache_data(ttl=86400, show_spinner=False)
def posterfetcher(web_movie_id):
    url = f'https://api.themoviedb.org/3/movie/{web_movie_id}'
    response = make_request_with_retry(url, params={'api_key': TMDB_API_KEY})
    data = response.json()
    return "https://image.tmdb.org/t/p/w500"+ data['poster_path']





# Recommender Function

def recommend(option_selected):

    index_movie = df_import_list[df_import_list['title']== option_selected].index[0]

    # index_movie = df_import_list.index(curr_movie)   funtion related to the list

    currmovie_all_dist = Similarity[index_movie]

    top_recommendation = sorted(enumerate(currmovie_all_dist),reverse=True,key = lambda x:x[1])[1:6]

    # print (top_recommendation)

    output_list=[]

    output_poster=[]

    for i in top_recommendation:

        # fetch recomendation from data

        output_list.append(df_import_list.iloc[i[0]].title)
        # print(output_list)

        # fetch poster form api   

        web_movie_id = df_import_list.iloc[i[0]].movie_id

        output_poster.append(posterfetcher(web_movie_id))
        # print(output_poster)
    return output_list, output_poster




# page header

header = """

    # 🎥 Movie Recommender System
"""

# st.markdown(header, unsafe_allow_html=True)

st.subheader("", divider='rainbow')



# Select Box Discription

selectbox_discription = """

### Select a movie to get recommendations :"""

st.markdown(selectbox_discription, unsafe_allow_html=True)



# Select Box

def custom_css(css: str):

    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
custom_css('''

    div[data-baseweb="select"] > div {

        height: 60px;

    }
    

    div[data-baseweb="select"] > div > div {

        height: 60px;

        font-size: 30px;

    }
    

    div[data-baseweb="select"] input {

        height: 60px;

        font-size: 16px;

    }

''')#search box main body 2. search box text alignment and padding 3. search box input text size and padding

option_selected = st.selectbox('', df_import_list['title'].values, index=None, placeholder='Choose an option or type here',label_visibility='collapsed')



# Recommender Button


# other way to make the button in center

#col1, col2, col3 = st.columns([1.8,1,1.5])

#there is no conscept of width in streamlit so we use beta columns to make the page divided by columns  and then place the button in center column


if st.button('Recommend'):#change st to col2 (otherway)

    if option_selected:

        movie_name,poster = recommend(option_selected)# function call 


        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            st.image(poster[0])   
            # st.markdown(f"<img src='{poster[0]}' style='border-radius: 5%;'>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center;'>{movie_name[0]}</div>", unsafe_allow_html=True)
            # st.write(movie_name[0])
        

        with col2:

            st.image(poster[1])
            st.markdown(f"<div style='text-align: center;'>{movie_name[1]}</div>", unsafe_allow_html=True)
            # st.text(movie_name[1] )

        with col3:
            

            st.image(poster[2])
            st.markdown(f"<div style='text-align: center;'>{movie_name[2]}</div>", unsafe_allow_html=True)
            # st.text(movie_name[2])

        with col4:
            

            st.image(poster[3])
            st.markdown(f"<div style='text-align: center;'>{movie_name[3]}</div>", unsafe_allow_html=True)
            # st.text(movie_name[3])

        with col5:


            st.image(poster[4])
            st.markdown(f"<div style='text-align: center;'>{movie_name[4]}</div>", unsafe_allow_html=True)
            # st.text(movie_name[4])

    else:

        st.warning('Input is empty. Please enter a movie name.')



# sidebar

with st.sidebar:

    styled_title = "<h1 style='font-size: 30px; text-align: center; margin-top: 0;'>Watch Later</h1>"

    st.markdown(styled_title, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(label='', type=['jpeg', 'png'], accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #   img_data = uploaded_file.read()
    #   st.image(img_data, caption=uploaded_file.name, use_column_width=True)
  
  
  
    columns = st.columns(2)
    for i, uploaded_file in enumerate(uploaded_files):
        img_data = Image.open(uploaded_file)
        # Display the image in the corresponding column
        columns[i % 2].image(img_data, use_column_width=True)


# remove the full screen option     
# drag drop using button 
# hamburgur menu    

