import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import io

# Fungsi untuk mengekstrak teks dari file PDF yang diunggah
def extract_text_from_pdf(pdf_file):
    """
    Membaca file PDF yang diunggah dan mengekstrak teks dari semua halaman.
    """
    try:
        # Membaca file dalam memori
        pdf_reader = PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        # Loop melalui setiap halaman dan tambahkan teksnya
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file PDF: {e}")
        return None

# Fungsi untuk menganalisis teks kontrak menggunakan API OpenAI
def analyze_contract_with_openai(api_key, contract_text):
    """
    Mengirimkan teks kontrak ke OpenAI API untuk dianalisis.
    """
    # Validasi API Key
    if not api_key.startswith('sk-'):
        st.error("Format OpenAI API Key tidak valid. Pastikan diawali dengan 'sk-'.")
        return None

    try:
        # Inisialisasi client OpenAI dengan API key
        client = OpenAI(api_key=api_key)
        
        # Prompt yang dirancang untuk menganalisis dokumen kontrak
        prompt_message = f"""
        Anda adalah seorang asisten hukum AI yang ahli dalam menganalisis dokumen kontrak.
        Tolong analisis teks dokumen kontrak berikut dan berikan ringkasan terstruktur dalam format Markdown.
        Fokus pada poin-poin kunci berikut:

        1.  **Ringkasan Umum**: Jelaskan secara singkat tujuan dari kontrak ini.
        2.  **Para Pihak**: Identifikasi semua pihak yang terlibat dalam kontrak (misalnya, PIHAK PERTAMA, PIHAK KEDUA, nama perusahaan, atau individu).
        3.  **Tanggal-Tanggal Penting**: Sebutkan tanggal efektif, tanggal berakhir, atau tanggal penting lainnya yang disebutkan.
        4.  **Kewajiban Utama**: Rangkum kewajiban utama dari masing-masing pihak.
        5.  **Klausul Penting**: Identifikasi klausul-klausul penting seperti klausul kerahasiaan, ganti rugi, pemutusan kontrak, dan yurisdiksi hukum.
        6.  **Potensi Risiko atau Peringatan**: Tandai area mana pun yang mungkin ambigu, berisiko, atau memerlukan perhatian lebih lanjut.

        Berikut adalah teks kontraknya:
        ---
        {contract_text}
        ---
        """

        # Melakukan panggilan ke API
        response = client.chat.completions.create(
            model="gpt-4o",  # Atau model lain seperti "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "Anda adalah asisten hukum AI yang sangat teliti."},
                {"role": "user", "content": prompt_message}
            ],
            temperature=0.3, # Mengurangi kreativitas untuk hasil yang lebih faktual
            max_tokens=1500  # Menambah batas token untuk dokumen yang panjang
        )
        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghubungi API OpenAI: {e}")
        return None

# --- UI Aplikasi Streamlit ---

# Judul utama aplikasi
st.title("📄 Analisis Dokumen Kontrak dengan AI")
st.write("Unggah dokumen kontrak Anda dalam format PDF untuk mendapatkan ringkasan dan analisis poin-poin kunci secara otomatis.")

# Sidebar untuk input API Key
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    # Input API Key OpenAI dengan tipe password agar tidak terlihat
    openai_api_key = st.text_input(
        "Masukkan OpenAI API Key Anda", 
        type="password",
        help="Dapatkan API key Anda dari platform.openai.com"
    )
    st.markdown("---")
    st.info("API Key Anda tidak disimpan dan hanya digunakan untuk sesi ini.")

# Komponen untuk mengunggah file
uploaded_file = st.file_uploader(
    "Pilih file kontrak (PDF)", 
    type="pdf"
)

# Tombol untuk memulai analisis
if st.button("Analisa Dokumen"):
    # Validasi sebelum memulai
    if not openai_api_key:
        st.warning("Silakan masukkan OpenAI API Key Anda di sidebar terlebih dahulu.")
    elif uploaded_file is None:
        st.warning("Silakan unggah file PDF kontrak terlebih dahulu.")
    else:
        # Proses analisis jika semua sudah siap
        with st.spinner("Harap tunggu, AI sedang membaca dan menganalisis dokumen..."):
            # 1. Ekstrak teks dari PDF
            contract_text = extract_text_from_pdf(uploaded_file)
            
            if contract_text:
                st.info("✅ Teks berhasil diekstrak dari PDF. Mengirim ke AI untuk dianalisis...")
                # 2. Kirim teks ke OpenAI untuk dianalisis
                analysis_result = analyze_contract_with_openai(openai_api_key, contract_text)
                
                # 3. Tampilkan hasil
                if analysis_result:
                    st.subheader("Hasil Analisis Kontrak")
                    st.markdown(analysis_result)
                else:
                    st.error("Gagal mendapatkan hasil analisis. Silakan periksa API Key Anda dan coba lagi.")
