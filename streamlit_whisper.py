import asyncio
import streamlit as st
import whisper
import os
import tempfile

# 🔧 Correction du bug "no running event loop" (asyncio)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 🛠️ Désactiver le mode "watch" de Streamlit pour éviter les erreurs de redémarrage
st.set_option('server.runOnSave', False)

# ✅ Vérification de Streamlit
st.write("✅ Streamlit fonctionne bien !")

# 🚀 Chargement du modèle Whisper
@st.cache_resource
def load_model():
    return whisper.load_model("tiny").to("cpu")  # Test avec le modèle "tiny" sur CPU

# Charger le modèle
model = load_model()
st.write("✅ Modèle Whisper chargé avec succès !")

# 🎙️ Interface utilisateur
st.title("🎙️ Transcription Audio en Texte")
st.write("Déposez votre fichier audio pour obtenir une transcription en texte.")

# 📂 Upload du fichier audio
uploaded_file = st.file_uploader("Choisissez un fichier audio (MP3, WAV, M4A, etc.)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # 🔧 Création d'un fichier temporaire pour l'upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        file_path = temp_file.name

    st.success("✅ Fichier bien reçu ! Début de la transcription...")

    # 🎤 Transcrire l'audio
    try:
        resultat = model.transcribe(file_path, language="fr")
        transcription = resultat["text"]

        # 📄 Affichage du texte transcrit
        st.subheader("📝 Résultat de la transcription :")
        st.text_area("Texte transcrit", transcription, height=300)

        # 📥 Option pour télécharger la transcription
        st.download_button(
            label="📥 Télécharger la transcription",
            data=transcription,
            file_name="transcription.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")

    # 🧹 Nettoyage du fichier temporaire
    os.remove(file_path)
