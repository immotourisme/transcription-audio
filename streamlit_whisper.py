import streamlit as st
import whisper
import os

# Test pour voir si Streamlit fonctionne avant le chargement du modèle
st.write("✅ Streamlit fonctionne bien !")

@st.cache_resource
def load_model():
    return whisper.load_model("base").to("cpu")  # Force l'utilisation du CPU

# Charger le modèle Whisper
model = load_model()

st.write("✅ Modèle Whisper chargé avec succès !")

# Interface utilisateur Streamlit
st.title("🎙️ Transcription Audio en Texte")
st.write("Déposez votre fichier audio pour obtenir une transcription en texte.")

# Upload du fichier audio
uploaded_file = st.file_uploader("Choisissez un fichier audio (MP3, WAV, M4A, etc.)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Sauvegarder le fichier temporairement
    file_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("✅ Fichier bien reçu ! Début de la transcription...")
    
    # Transcrire l'audio
    resultat = model.transcribe(file_path, language="fr")
    transcription = resultat["text"]
    
    # Affichage du texte transcrit
    st.subheader("📝 Résultat de la transcription :")
    st.text_area("Texte transcrit", transcription, height=300)
    
    # Option pour télécharger la transcription
    transcription_file = file_path.replace(".m4a", "_transcription.txt")
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    st.download_button(
        label="📥 Télécharger la transcription",
        data=transcription,
        file_name="transcription.txt",
        mime="text/plain"
    )
    
    # Nettoyage du fichier temporaire
    os.remove(file_path)
