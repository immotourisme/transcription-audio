import os
import streamlit as st
import whisper
import tempfile
import ffmpeg
import shutil

# Vérification de FFmpeg
if not shutil.which("ffmpeg"):
    st.error("❌ FFmpeg n'est pas installé. Merci de l'ajouter aux dépendances.")
    st.stop()

# Fonction de conversion en WAV pour éviter les erreurs de format
def convert_to_wav(input_path):
    output_path = input_path.rsplit(".", 1)[0] + ".wav"
    try:
        ffmpeg.input(input_path).output(output_path, format="wav").run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return output_path
    except Exception as e:
        st.error(f"❌ Erreur lors de la conversion du fichier audio : {e}")
        return None

# Fonction de transcription
def transcrire_audio(fichier_audio, modele="medium", langue="fr"):
    """
    Transcrit un fichier audio en texte avec Whisper.
    """
    # Chargement du modèle Whisper
    st.info(f"🔄 Chargement du modèle Whisper ({modele})...")
    model = whisper.load_model(modele)

    # Transcription
    st.info("🎙️ Transcription en cours... Cela peut prendre plusieurs minutes.")
    resultat = model.transcribe(fichier_audio, language=langue)

    return resultat["text"]

# Interface utilisateur Streamlit
st.title("🎙️ Transcription Audio en Texte")
st.write("Déposez votre fichier audio pour obtenir une transcription en texte.")

# Upload du fichier
uploaded_file = st.file_uploader("Choisissez un fichier audio (MP3, WAV, M4A, etc.)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        file_path = temp_file.name

    st.success("✅ Fichier bien reçu ! Début de la transcription...")
    st.warning("⚠️ Attention : La transcription peut prendre plusieurs minutes en fonction de la taille du fichier. \n\nExemple : 15/20 minutes pour un fichier de 30 Mo (soit 1h de discussion). Merci de bien vouloir garder cette page ouverte.")

    # Conversion en WAV
    file_path = convert_to_wav(file_path)
    if file_path is None:
        st.stop()

    # Lancer la transcription
    try:
        transcription = transcrire_audio(file_path, modele="medium", langue="fr")

        # Affichage du texte
        st.subheader("📝 Résultat de la transcription :")
        st.text_area("Texte transcrit", transcription, height=300)

        # Option pour télécharger le fichier
        st.download_button(
            label="📥 Télécharger la transcription",
            data=transcription,
            file_name="transcription.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")

    # Nettoyage des fichiers temporaires
    os.remove(file_path)
