import streamlit as st
import openai
from PyPDF2 import PdfReader
import io

# Judul aplikasi
st.set_page_config(page_title="Penganalisis Kontrak AI")
st.title("Penganalisis Kontrak AI")
st.write("Unggah dokumen kontrak PDF Anda dan biarkan AI menganalisisnya.")

# Sidebar untuk API Key
st.sidebar.header("Konfigurasi API")
api_key = st.sidebar.text_input("Masukkan OpenAI API Key Anda", type="password")

if api_key:
    openai.api_key = api_key
else:
    st.sidebar.warning("Silakan masukkan OpenAI API Key Anda untuk melanjutkan.")
    st.stop() # Hentikan eksekusi jika API Key belum dimasukkan

# Fungsi untuk mengekstrak teks dari PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Fungsi untuk menganalisis teks menggunakan OpenAI
def analyze_contract_with_openai(text, prompt_instruction):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Anda bisa coba model lain seperti "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "Anda adalah asisten yang ahli dalam menganalisis dokumen hukum, khususnya kontrak."},
                {"role": "user", "content": f"{prompt_instruction}\n\nKontrak:\n{text}"}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        st.error(f"Terjadi kesalahan API OpenAI: {e}")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return None

# Bagian utama aplikasi
st.header("Unggah Dokumen Kontrak Anda")
uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' berhasil diunggah.")

    # Ekstrak teks
    with st.spinner("Mengekstrak teks dari PDF..."):
        contract_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Pratinjau Teks Kontrak (2000 karakter pertama):")
    st.text_area("Teks yang diekstrak", contract_text[:2000] + "..." if len(contract_text) > 2000 else contract_text, height=300)

    st.header("Pilih Jenis Analisis")
    analysis_type = st.radio(
        "Apa yang ingin Anda ketahui tentang kontrak ini?",
        ("Ringkasan Utama", "Klausul Penting", "Risiko Potensial", "Pertanyaan Kustom")
    )

    if analysis_type == "Ringkasan Utama":
        prompt = "Berikan ringkasan singkat dan komprehensif dari kontrak ini, soroti tujuan utamanya dan para pihak yang terlibat."
    elif analysis_type == "Klausul Penting":
        prompt = "Identifikasi dan jelaskan 5 klausul paling penting dalam kontrak ini. Sertakan nomor pasal atau bagian jika ada."
    elif analysis_type == "Risiko Potensial":
        prompt = "Analisis kontrak ini untuk mengidentifikasi potensi risiko atau kerugian bagi salah satu pihak. Berikan rekomendasi mitigasi jika memungkinkan."
    elif analysis_type == "Pertanyaan Kustom":
        st.subheader("Ajukan Pertanyaan Anda")
        custom_question = st.text_area("Tulis pertanyaan spesifik Anda tentang kontrak ini:", height=100)
        if custom_question:
            prompt = custom_question
        else:
            st.warning("Silakan masukkan pertanyaan kustom Anda.")
            st.stop()

    if st.button("Mulai Analisis"):
        if contract_text:
            with st.spinner("Menganalisis kontrak dengan AI... Ini mungkin memakan waktu beberapa saat."):
                analysis_result = analyze_contract_with_openai(contract_text, prompt)
            
            if analysis_result:
                st.subheader("Hasil Analisis:")
                st.markdown(analysis_result) # Gunakan markdown untuk menampilkan hasil dengan format yang lebih baik
        else:
            st.error("Tidak ada teks yang diekstrak dari dokumen PDF.")
