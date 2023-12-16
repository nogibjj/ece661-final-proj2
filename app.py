import torch
# import streamlit as st
from PIL import Image
from data.dataset import ClassifierDataset,TransformerDatasetREMI
import random
from config import *
from transformer_generator import *
from torch.nn.functional import softmax
from data.process_data import MIDIEncoderREMI
import os
import streamlit as st
import pretty_midi
from scipy.io import wavfile
import numpy as np
import io
import muspy
# import fluidsynth

max_seq_len = 256
single_file_dataset_path = "data/single_file_dataset.npz"
classifier_dataset = ClassifierDataset(single_file_dataset_path, seq_len=max_seq_len, labels_path="data/emopia/EMOPIA_2.2/label.csv")
generator_dataset = TransformerDatasetREMI(single_file_dataset_path, seq_len=max_seq_len)

Q1, Q2, Q3, Q4 = [], [], [], []
for dic in classifier_dataset:
    label = dic['target']
    if label == 0:
        Q1.append(dic)
    elif label == 1:
        Q2.append(dic)
    elif label == 2:
        Q3.append(dic)
    elif label == 3:
        Q4.append(dic)

model1.load_state_dict(torch.load('checkpoints/transformer_v3.pt'))
model2.load_state_dict(torch.load('checkpoints/transformer.pt'))
model3.load_state_dict(torch.load('checkpoints/generator_c.pt'))

def generate(emotion = None):
    # Generate music based on the selected emotion

    if emotion is None:
        data = random.choice(generator_dataset)
        input = data['input'].to(device)
        target = data['target'].to(device)
    else:
        if emotion == 'Happy':
            dic = random.choice(Q1)
        elif emotion == 'Stressed':
            dic = random.choice(Q2)
        elif emotion == 'Sad':
            dic = random.choice(Q3)
        elif emotion == 'Peaceful':
            dic = random.choice(Q4)
        input = dic['input'].to(device)
        target = torch.cat((input[1:], torch.tensor([0],dtype=torch.long).to(device)))

    model1.eval()
    model2.eval()
    model3.eval()

    # current_token = start_token
    generated_musics = {'model1':[], 'model2':[], 'model3':[]}

    with torch.no_grad():
        generated_musics['model1'].append(input[0])
        generated_musics['model2'].append(input[0])
        generated_musics['model3'].append(input[0])

        output1 = model1(input)
        output2 = model2(input.unsqueeze(0), target.unsqueeze(0))
        output3 = model3(input.unsqueeze(0))
        # Apply temperature to the output probabilities for diversity

        probabilities1 = softmax(output1.squeeze() / TEMPERATURE, dim=-1)
        probabilities2 = softmax(output2.squeeze() / TEMPERATURE, dim=-1)
        probabilities3 = softmax(output3.squeeze() / TEMPERATURE, dim=-1)

        for j in range(MAX_SEQ_LEN):
            current_token1 = torch.multinomial(probabilities1[j], 1).item()
            if current_token1 == END_TOKEN:
                break
            else:
                generated_musics['model1'].append(current_token1)
        for j in range(MAX_SEQ_LEN):
            current_token2 = torch.multinomial(probabilities2[j], 1).item()
            if current_token2 == END_TOKEN:
                break
            else:
                generated_musics['model2'].append(current_token2)
        for j in range(MAX_SEQ_LEN):
            current_token3 = torch.multinomial(probabilities3[j], 1).item()
            if current_token3 == END_TOKEN:
                break
            else:
                generated_musics['model3'].append(current_token3)
    return generated_musics
    


def main():
    st.title("Piano Music Generator")
    st.write("This is a music generator based on the emotions.")
   
    image = Image.open('piano.jpg')
    st.image(image)

    st.write("Select the emotion for your music.")

    emotion_list = ['Happy', 'Stressed', 'Sad', 'Peaceful']
    selected_emo = st.selectbox( "Type or select a year from the dropdown", emotion_list)

    if selected_emo == 'Happy':
        original_music = random.choice(Q1)['input'].numpy()
    elif selected_emo == 'Stressed':
        original_music = random.choice(Q2)['input'].numpy()
    elif selected_emo == 'Sad':
        original_music = random.choice(Q3)['input'].numpy()
    else:
        original_music = random.choice(Q4)['input'].numpy()
    
    if st.button('Generate'):
        generated_musics = generate(selected_emo)
        # Instantiate your MidiEncoder and MidiEncoderREMI
        path_to_midi = "data/emopia/EMOPIA_2.2/midis/"
        midi_files_list = [os.path.join(path_to_midi, file) for file in os.listdir(path_to_midi) if file.endswith(".mid")]
        midi_encoder_remi = MIDIEncoderREMI(dict_path="data/encoder_dict.pkl", midi_files_list=midi_files_list)
        for key in generated_musics.keys():
            midi_encoder_remi.words_to_midi(generated_musics[key],f'presentation/{key}.mid')
        midi_encoder_remi.words_to_midi(original_music,f'presentation/original.mid')

        with st.spinner(f"Transcribing to FluidSynth"):
            # for key in generated_musics.keys():

            midi_file = 'presentation/original.mid'
            music =  muspy.read_midi(midi_file)
            plotter = muspy.show_score(music, figsize=(10, 6), clef="treble", clef_octave=0)
            fig = plotter.fig
            fig.savefig('presentation/original.png')
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            audio_data = midi_data.fluidsynth()
            audio_data = np.int16(
                audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
            )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

            virtualfile0 = io.BytesIO()
            wavfile.write(virtualfile0, 44100, audio_data)

            midi_file = 'presentation/model1.mid'
            music =  muspy.read_midi(midi_file)
            plotter = muspy.show_score(music, figsize=(10, 6), clef="treble", clef_octave=0)
            fig = plotter.fig
            fig.savefig('presentation/model1.png')
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            audio_data = midi_data.fluidsynth()
            audio_data = np.int16(
                audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
            )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

            virtualfile1 = io.BytesIO()
            wavfile.write(virtualfile1, 44100, audio_data)

            midi_file = 'presentation/model2.mid'
            music =  muspy.read_midi(midi_file)
            plotter = muspy.show_score(music, figsize=(10, 6), clef="treble", clef_octave=0)
            fig = plotter.fig
            fig.savefig('presentation/model2.png')
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            audio_data = midi_data.fluidsynth()
            audio_data = np.int16(
                audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
            )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

            virtualfile2 = io.BytesIO()
            wavfile.write(virtualfile2, 44100, audio_data)

            midi_file = 'presentation/model3.mid'
            music =  muspy.read_midi(midi_file)
            plotter = muspy.show_score(music, figsize=(10, 6), clef="treble", clef_octave=0)
            fig = plotter.fig
            fig.savefig('presentation/model3.png')
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            audio_data = midi_data.fluidsynth()
            audio_data = np.int16(
                audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
            )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

            virtualfile3 = io.BytesIO()
            wavfile.write(virtualfile3, 44100, audio_data)
        st.text("Original Music")
        st.image(Image.open('presentation/original.png'))
        st.audio(virtualfile0)
        
        st.text("Generated Music by Transformer Decoder Model")
        st.image(Image.open('presentation/model1.png'))
        st.audio(virtualfile1)

        st.text("Generated Music by Transformer Encoder Decoder Model")
        st.image(Image.open('presentation/model3.png'))
        st.audio(virtualfile3)

        st.text("Generated Music by Transformer Gan Model")
        st.image(Image.open('presentation/model2.png'))
        st.audio(virtualfile2)

    # menu = ["Home", "Choose Your Song By Singer","Choose Your Song By Song Name","Interaction with the dataset"]
    # choice = st.sidebar.selectbox("Menu", menu)
    # if choice == "Home":
    #     st.subheader("Home")
    #     st.write("This is the home page.")
    # elif choice == "Choose Your Song By Singer":
    #     st.subheader("Recommendation")
    #     st.write("Please enter the singer name and choose song name.")
    #     artitst_list = df.toPandas()['artists'].unique()
    #     selected_singer = st.selectbox( "Type or select an artist from the dropdown", artitst_list)
    #     song_list = df.filter(df.artists.contains(selected_singer)).toPandas()['name'].values
    #     selected_song = st.selectbox( "Type or select a song from the dropdown", song_list)
    #     num_of_songs = st.slider("Number of songs to recommend", 1, 30, 5)
    #     if st.button("Recommend"):
    #         recommendation = get_distance(selected_song,num_of_songs,selected_singer)
    #         st.write(recommendation)
    # elif choice == "Choose Your Song By Song Name":
    #     st.subheader("Recommendation")
    #     st.write("This is the recommendation page.")
    #     song_list = df.toPandas()['name'].unique()
    #     selected_song = st.selectbox( "Type or select a song from the dropdown", song_list)
    #     artitst_list = df.filter(df.name==selected_song).toPandas()['artists'].values
    #     selected_artist = st.selectbox( "Type or select an artist from the dropdown", artitst_list)
    #     num_of_songs = st.slider("Number of songs to recommend", 1, 30, 5)
    #     if st.button('Show Recommendation'):
    #         if selected_song is not None and selected_artist is not None:
    #             recommended_song_names = get_distance(selected_song,num_of_songs,selected_artist)
    #             st.dataframe(recommended_song_names)

    # elif choice == "Interaction with the dataset":
    #     st.subheader("Here are some interactions with the dataset")
    #     # popularity of an artist over the years
    #     st.write("Popularity of an artist over the years")
    #     artitst_list = df.toPandas()['artists'].unique()
    #     selected_singer = st.selectbox( "Type or select an artist from the dropdown", artitst_list)
    #     # df.filter(df.location.contains('google.com'))
    #     artist_data = df.filter(df.artists.contains(selected_singer)).toPandas().groupby("year").mean()
    #     fig = go.Figure([go.Scatter(x=artist_data.index, y=artist_data["popularity"])],layout_title_text="Popularity of "+selected_singer+" over the years")
    #     st.plotly_chart(fig, use_container_width=True)

    #     st.write("Top artists ranking for a year")
    #     # Artist ranking for a year
    #     year_list = sorted(df.toPandas()['year'].unique(),reverse=True)
    #     selected_year = st.selectbox( "Type or select a year from the dropdown", year_list,key = 'Artist ranking for a year')

    #     num_of_artists = st.slider("Number of artists to show", 1, 30, 15)

    #     artist_ranking_year = df.filter(df.year == str(selected_year)).toPandas()
    #     artist_ranking_year = artist_ranking_year.groupby("artists").mean().sort_values(["popularity"],ascending=False).head(num_of_artists)
    #     fig = go.Figure(
    #         data=[go.Bar(x=artist_ranking_year.index,y=artist_ranking_year["popularity"])],
    #         layout_title_text="Artist ranking for the year - "+str(selected_year))
    #     st.plotly_chart(fig, use_container_width=True)

    #     # Popular artist ranking along with the number of hit songs released over the years

    #     num_of_artists = st.slider("Number of artists to show", 1, 50, 25,key = 'Popular artist ranking along with the number of hit songs released over the years')
    #     st.write("Artist ranking along with the number of hit songs released over the years")
    #     artist_ranking_hit = df.toPandas().groupby("artists").mean().sort_values(["popularity"],ascending=False).head(num_of_artists)
    #     artist_ranking_hit["artists"] = artist_ranking_hit.index.values
    #     artist_ranking_hit["count"] = df.toPandas().groupby("artists").count()["popularity"].head(num_of_artists).values

    #     fig = make_subplots(specs=[[{"secondary_y": True}]])

    #     fig.add_trace(go.Bar(x=artist_ranking_hit["artists"],
    #                     y=artist_ranking_hit["popularity"],
    #                     text = artist_ranking_hit["count"],
    #                     hovertemplate = '<b>Artist: </b>%{x}<br><b>Popularity: </b>%{y:.2f}<br><b> # Songs: </b>%{text}',
    #                     showlegend = False
    #                     ),
    #         secondary_y=False,
    #     )
    #     fig.add_trace(go.Scatter(
    #         x = artist_ranking_hit["artists"],
    #         y = artist_ranking_hit["count"],
    #         text = artist_ranking_hit["popularity"],
    #         hovertemplate = '<b>Artist: </b>%{x}<br><b>Popularity: </b>%{text:.2f}<br><b> # Songs: </b>%{y}',
    #         showlegend = False),
    #         secondary_y=True,)

    #     # Set x-axis title
    #     fig.update_xaxes(title_text="Artist ranking along with the number of songs released over the years")

    #     # Set y-axes titles
    #     fig.update_yaxes(title_text="<b>Popularity</b>", secondary_y=False)
    #     fig.update_yaxes(title_text="<b># Hit songs", secondary_y=True)

    #     st.plotly_chart(fig)

if __name__ == '__main__':
    main()