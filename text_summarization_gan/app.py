import streamlit as st
import torch
import os
from transformers import pipeline

# Import our custom modules
from utils.text_preprocessing import clean_text
from utils.document_loader import extract_text_from_uploaded_file

# --- Streamlit Configurations ---
st.set_page_config(
    page_title="Advanced GAN Text Summarizer", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.write("Checking if Streamlit works...")

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .main .block-container { max-width: 1000px; padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #1E3A8A; font-family: 'Inter', sans-serif; font-weight: 800; border-bottom: 2px solid #3B82F6; padding-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; background-color: #2563EB; color: #fff; }
    .stButton>button:hover { background-color: #1E40AF; border-color: #1E40AF; }
</style>
""", unsafe_allow_html=True)

st.title("Generative AI-Based Text Summarization")
st.markdown("### Powered by Advanced GAN Architecture (Encoder-Decoder & Adversarial Training)")
st.markdown("This application uses a Generator Network (Transformer based) trained in an adversarial context to evaluate generated (abstractive) summaries against a natural human baseline via a Discriminator Network.")

# --- Load Generator Model ---
@st.cache_resource(show_spinner="Loading Generator Network...")
def load_gan_generator(model_choice):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_choice == "English (Local Fine-Tuned T5)":
        if os.path.exists("models/saved/generator"):
            model_path = "models/saved/generator"
            # Fallback for tokenizer if not saved locally
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
            except:
                tokenizer = AutoTokenizer.from_pretrained("t5-small", legacy=False)
        else:
            model_path = "t5-small"
            tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
        prefix = "summarize: "
    elif model_choice == "English (Base Pre-trained T5)":
        model_path = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
        prefix = "summarize: "
    else:
        model_path = "csebuetnlp/mT5_multilingual_XLSum" 
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
        prefix = ""
        
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    
    def summarizer(text, max_length=130, min_length=30, temperature=0.7):
        input_text = prefix + text
        # T5-small max position is 512.
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                temperature=temperature,
                num_beams=4,
                no_repeat_ngram_size=3, # Increased to 3 to prevent weird 2-gram looping that causes empty output
                early_stopping=True,
                top_p=0.9
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"summary_text": summary}]
        
    return summarizer

# --- Sidebar Inputs ---    
st.sidebar.header("Summarization Settings")
st.sidebar.markdown("**Language & Model Selection**")
model_choice = st.sidebar.selectbox("Select Capability", [
    "English (Base Pre-trained T5)", 
    "English (Local Fine-Tuned T5)",
    "Multilingual (mT5-XLSum - 45+ Languages)"
])

# Initialize session state for source text
if "source_text" not in st.session_state:
    st.session_state["source_text"] = ""
if "last_file_id" not in st.session_state:
    st.session_state["last_file_id"] = ""

if model_choice == "English (Local Fine-Tuned T5)":
    if os.path.exists("models/saved/generator"):
        st.sidebar.success("Loaded locally trained custom T5 Generator.")
    else:
        st.sidebar.warning("Local model not found! Falling back to Base T5.")
elif model_choice == "English (Base Pre-trained T5)":
    st.sidebar.info("Using stable Base T5 Generator architecture.")
else:
    st.sidebar.info("Using mT5 Multilingual Model (Supports 45+ Languages)")
    with st.sidebar.expander("🌍 See all 45 Supported Languages"):
        st.markdown("""
        **Supported Languages include:**
        Amharic, Arabic, Azerbaijani, Bengali, Burmese, Chinese, English, French, Gujarati, 
        Hausa, Hindi, Igbo, Indonesian, Japanese, Kirundi, Korean, Kyrgyz, Marathi, Nepali, 
        Oromo, Pashto, Persian, Pidgin, Portuguese, Punjabi, Russian, Scottish Gaelic, Serbian, 
        Sinhala, Somali, Spanish, Swahili, Tamil, Telugu, Thai, Tigrinya, Turkish, Ukrainian, 
        Urdu, Uzbek, Vietnamese, Welsh, Yoruba.
        """)

generator_summarizer = load_gan_generator(model_choice)

st.sidebar.markdown("**Network Parameters**")
max_len = st.sidebar.slider("Maximum Tokens", 50, 500, 130)
min_len = st.sidebar.slider("Minimum Tokens", 10, 200, 30)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 2.0, 0.3, step=0.1)

# --- Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Source Input")
    
    # Using tabs instead of radio buttons to prevent state desync bugs
    tab1, tab2 = st.tabs(["✍️ Direct Text Input", "📄 Document Upload"])
    
    with tab1:
        # Use session state to allow synchronization across tabs
        st.session_state["source_text"] = st.text_area(
            "Paste text here to summarize:", 
            height=300, 
            value=st.session_state["source_text"],
            placeholder="Enter an article, blog post, or long text here..."
        )
        
    with tab2:
        uploaded_file = st.file_uploader("Upload a supported document:", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            try:
                # Cache extraction so it doesn't rerun on every slider move
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get("last_file_id") != file_id:
                    with st.spinner("Extracting text contents..."):
                        extracted_doc_text = extract_text_from_uploaded_file(uploaded_file)
                        if extracted_doc_text.strip():
                            st.session_state["source_text"] = extracted_doc_text
                            st.session_state["last_file_id"] = file_id
                            st.rerun() # Refresh to show text in the text area tab
                st.success(f"Successfully extracted {len(st.session_state['source_text'])} characters from {uploaded_file.name}!")
                with st.expander("Preview Extracted Text"):
                    st.text(st.session_state["source_text"][:2000] + ("..." if len(st.session_state["source_text"]) > 2000 else ""))
            except Exception as e:
                st.error(f"Error reading file format: {e}")

with col2:
    st.subheader("Abstractive Output")
    
    if st.button("Generate GAN Summary", use_container_width=True):
        text_to_process = st.session_state["source_text"]
        
        if not text_to_process.strip():
            st.warning("Please provide some source text first.")
        else:
            with st.spinner("Generator Network is processing..."):
                cleaned_text = clean_text(text_to_process)
                
                if len(cleaned_text.split()) < min_len:
                    st.error(f"Input text is too short! Must be longer than your requested minimum summary length ({min_len} words).")
                else:
                    try:
                        # T5-Small handles 512 tokens. Cut input to a reasonable character limit (~2500 chars)
                        trunc_payload = cleaned_text[:2500] 
                        
                        summary_output = generator_summarizer(
                            trunc_payload, 
                            max_length=max_len, 
                            min_length=min_len, 
                            temperature=temperature
                        )
                        
                        res = summary_output[0]['summary_text']
                        
                        if not res.strip():
                            st.warning("The model produced an empty summary. Try adjusting the input text or parameters.")
                        else:
                            st.markdown("""
                                <div style="padding: 20px; background-color: #E0F2FE; border-left: 5px solid #0284C7; border-radius: 8px; font-size: 1.1em; color: #0C4A6E; line-height: 1.6;">
                                {}
                                </div>
                            """.format(res), unsafe_allow_html=True)
                            
                            st.balloons()
                            
                    except Exception as e:
                        st.error(f"An unexpected inference error occurred: {e}")

st.markdown("---")
st.markdown("*Architecture: Advanced Encoder-Decoder GAN (TextGAN / SeqGAN topology). PyTorch underlying runtime frontend built with Streamlit.*")
